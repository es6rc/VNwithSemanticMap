from NewMethods.utils import helper
import pickle
get_starting_points = helper.get_starting_points
get_target_points = helper.get_target_points
get_minimal_steps = helper.get_minimal_steps

from Environment.navi_env_test import SmapHouse

house_id = '5cf0e1e9493994e483e985c436b9d3bc'
house = SmapHouse(env=house_id, sz=0)
all_targets = house.cat
starting_points = get_starting_points(house_id, all_targets)
target_points = get_target_points(house_id, all_targets)
# Category start poses
cat_startpoints = {cat:startpoint for cat, startpoint
                   in zip(all_targets, starting_points)}
# Category target poses
cat_targetpoints = {cat: target_point for cat, target_point
                    in zip(all_targets, target_points)}


cat_trajs = {}
states_trajs = {}
# cat_steps = {}
# states_steps = {}
for pos in house.map.keys():
    for orien in range(0, 4):
        state = (pos[0], pos[1], orien)
        for cat in all_targets:
            _, cat_trajs[cat] = get_minimal_steps(house_id, [state], [cat_targetpoints[cat]])
        # states_steps[state] = cat_steps
        states_trajs[state] = cat_trajs
        cat_trajs = {}

states_aseq = {}
for loc in states_trajs.keys():
    state_targettraj = states_trajs[loc]

    state_target_aseq = {}
    for target in state_targettraj.keys():
        traj = state_targettraj[target][0]
        target_aseq = []
        for i in range(len(traj)):
            #Abandon location info and also last few poses at the same location
            if (traj[i][1], traj[i][2]) != (traj[-1][1], traj[-1][2]):
                target_aseq.append(traj[i+1][0])
        target_aseq.append(-1)
        state_target_aseq[target] = target_aseq
    states_aseq[loc] = state_target_aseq

pickle.dump(states_aseq, open('%s/planner_data/states_aseq.pkl' % house.dir, 'wb'))
pickle.dump(house.fmap, open('%s/planner_data/fmap.pkl' % house.dir, 'wb'))
pickle.dump(cat_startpoints, open('%s/planner_data/startlocs.pkl' % house.dir, 'wb'))
pickle.dump(cat_targetpoints, open('%s/planner_data/targetlocs.pkl' % house.dir, 'wb'))
