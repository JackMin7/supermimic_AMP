"""AMP/recovery imagination with explicit task state and frozen reward networks."""
import torch
from .imagination import critic_input
from .numerics import guarded_action,valid_mean

TASK_KEYS=['command','heading_target','command_time_left','is_heading_env','is_standing_env',
           'recovery_enabled','recovery_counter','episode_step']
START_KEYS=['state','context','task/observation_history','task/previous_action']+['task/'+k for k in TASK_KEYS]


def value_input(wm,state,previous,context,command,task,heads):
    extra=torch.stack((task['recovery_enabled'].float(),task['recovery_counter']/max(heads.maximum,1),
                       task['episode_step']/heads.max_episode_steps,task['command_time_left']/8,
                       task['heading_target'].sin(),task['heading_target'].cos(),
                       task['is_heading_env'].float(),task['is_standing_env'].float()),-1)
    return torch.cat((critic_input(wm,state,previous,context,command),extra),-1)


def sample_starts(replay,ids,heads,batch=128,horizon=8):
    ages,envs=replay.valid_starts(horizon,ids); rows=replay.rows(ages)
    good=(replay.storage['task/command_time_left'][rows,envs]>horizon*heads.dt)&(replay.storage['task/previous_action'][rows,envs].abs().amax(-1)<10)
    ages,envs=ages[good],envs[good]; rows=replay.rows(ages)
    if not len(ages): raise ValueError('No P4 start windows within command lifetime')
    # Guarantee recovery data actually participates, without sampling held-out envs.
    recovery=replay.storage['task/recovery_enabled'][rows,envs]
    picks=[]
    for mask,count in ((recovery,int(.4*batch)),(~recovery,batch-int(.4*batch))):
        candidates=mask.nonzero().flatten()
        if not len(candidates): raise ValueError('Missing recovery or locomotion training partition')
        picks.append(candidates[torch.randint(len(candidates),(count,),device=replay.device)])
    chosen=torch.cat(picks)
    return replay.gather(ages[chosen],envs[chosen],1,START_KEYS)


def imagine(wm,actor,normalizer,critic,heads,amp,features,starts,horizon=8,gamma=.99,teacher=None,anchor_weight=1.):
    state=starts['state'][:,0]; history=starts['task/observation_history'][:,0]
    previous=starts['task/previous_action'][:,0]; context=starts['context'][:,0]
    task={k:starts['task/'+k][:,0] for k in TASK_KEYS}
    alive=torch.ones(len(state),device=state.device); objective=torch.zeros_like(alive); total=torch.zeros_like(alive)
    xs=[]; rewards=[]; continuations=[]; anchors=[]; modes=[]; style_scores=[]; valid_steps=[]
    for t in range(horizon):
        command=heads.command(state,task)
        xs.append(value_input(wm,state,previous,context,command,task,heads))
        obs=normalizer(history.flatten(1)); action=actor(obs)
        action,valid=guarded_action(action);valid_steps.append(valid)
        anchor=torch.zeros_like(alive)
        if teacher is not None:
            with torch.no_grad(): target=teacher(obs)
            anchor=(action-target).square().mean(-1)
        ns,contact=wm(state,action,context)
        next_task,delayed,terminated,timeout,cont=heads.advance(ns,task)
        # Smooth hard-termination surrogate; timeouts never incur death penalty.
        death=torch.where(timeout,torch.zeros_like(cont),1-cont)
        terms=heads.reward_terms(state,ns,action,previous,command,delayed,death,contact)
        task_reward=heads.dt*sum(terms.values())
        reward,score=amp.reward(features(state),features(ns),task_reward)
        objective+=gamma**t*alive*(reward-heads.dt*anchor_weight*anchor)
        total+=gamma**t*alive*reward
        alive=alive*cont
        rewards.append(reward); continuations.append(cont); anchors.append(anchor); modes.append(delayed.float()); style_scores.append(score)
        next_command=heads.command(ns,next_task)
        history=torch.cat((history[:,1:],heads.observation(ns,action,next_command)[:,None]),1)
        state,previous,task=ns,action,next_task
    command=heads.command(state,task)
    xs.append(value_input(wm,state,previous,context,command,task,heads))
    bootstrap=gamma**horizon*alive*critic(xs[-1])
    valid=torch.stack(valid_steps).all(0)
    return {'objective':valid_mean(objective+bootstrap,valid),'task_return':valid_mean(total,valid),'anchor_mse':valid_mean(torch.stack(anchors).mean(0),valid),
            'critic_inputs':torch.stack(xs),'rewards':torch.stack(rewards),'continuations':torch.stack(continuations),
            'valid_trajectories':valid,'imagined_rejected_fraction':(~valid).float().mean(),
            'recovery_mode_fraction':torch.stack(modes).mean(),'style_score':torch.stack(style_scores).mean()}
