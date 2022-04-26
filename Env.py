# Import routines

import numpy as np
import math
import random


# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(p,q) for p in range(m) for q in range(m) if p!=q or p==0]
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0 for _ in range(m+t+d)]
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+state[2]] = 1
        return state_encod


    # Use this function if you are using architecture-2 
    def state_encod_arch2(self, state, action):
        """convert the (state-action) into a vector so that it can be fed to the NN. 
            This method converts a given state-action pair into a vector format. 
            Hint: The vector is of size m + t + d + m + m."""
        state_encod = [0 for _ in range(m+t+d+m+m)]
        state_encod[state[0]] = 1
        state_encod[m+state[1]] = 1
        state_encod[m+t+ state[2]] = 1
        if (action[0] != 0):
            state_encod[m+t+d+ action[0]] = 1
        if (action[1] != 0):
            state_encod[m+t+d+m+ action[1]] = 1

        return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        lambdas = [2, 12, 4, 7, 8]
        # getting the request using poisson distribution and limiting the number of requests to 15
        requests = min(np.random.poisson(lambdas[location]), 15)
        
        possible_actions_idx = random.sample(range(1, (m-1)*m +1), requests)+[0] # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_idx]
        return possible_actions_idx,actions   



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        curr_loc = state[0] # get the current location
        pickup_loc = action[0] # get the pickup location
        drop_loc = action[1] # get the drop location
        curr_time = state[1] # get current time
        curr_day = state[2] # get current day
        reward= -C # assuming that by default driver refuses to take ride and action is (0,0)
        
        if (not (drop_loc == 0 and pickup_loc == 0)): # If the driver accepts the ride calculate reward
            # time taken from cab current location to pickup location
            transit_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            # Updating day and time after reaching pickup location
            new_time, new_day =self.get_updated_timeday(curr_time,curr_day,transit_time) 
            # time taken in taking passenger from one location to other
            ride_time = Time_matrix[pickup_loc][drop_loc][new_time][new_day]
            # calculate reward
            reward = (R * ride_time) - C * (transit_time + ride_time)
            
        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        # Initialize various times
        total_time   = 0
        transit_time = 0    # to go from current  location to pickup location
        wait_time    = 0    # in case driver chooses to refuse all requests
        ride_time    = 0    # from Pick-up to drop
        
        # Derive the current location, time, day and request locations
        curr_loc = state[0] # get the current location
        pickup_loc = action[0] # get the pickup location
        drop_loc = action[1] # get the drop location
        curr_time = state[1] # get current time
        curr_day = state[2] # get current day
        
        # Now the driver can refuse all requests, is already at pickup point or not at the pickup point
        if ((pickup_loc== 0) and (drop_loc == 0)): # Refuse all requests, so wait time is 1 unit, next location is current location
            wait_time = 1
            new_loc = curr_loc
        elif (curr_loc == pickup_loc): # driver is already at pickup point,so 0 transit time and wait.
            ride_time = Time_matrix[curr_loc][drop_loc][curr_time][curr_day]
            # new location is the drop location
            new_loc = drop_loc
        else: # Driver is not at pickup point and will need to reach there first
            transit_time = Time_matrix[curr_loc][pickup_loc][curr_time][curr_day]
            time_after_transit, day_after_transit = self.get_updated_timeday(curr_time, curr_day, transit_time)
            
            # The driver is now at the pickup point
            # Time taken to drop the passenger
            ride_time = Time_matrix[pickup_loc][drop_loc][time_after_transit][day_after_transit]
            new_loc  = drop_loc

        # Calculate total time as sum of all durations
        total_time = (wait_time + transit_time + ride_time)
        new_time, new_day = self.get_updated_timeday(curr_time, curr_day, total_time)
        
        # Construct next_state using the next_loc and the new time states.
        next_state = [new_loc, new_time, new_day]
        return next_state, total_time

    def get_updated_timeday(self, time, day, ride_duration):
            """
            Takes in the current time, day and journey ride duration and returns
            the state post that journey.
            """
            ride_duration = int(ride_duration)

            if (time + ride_duration) < 24: # No change in day
                time = time + ride_duration
            else: # Calculate the total time duration of subsequent days and convert time in 0-23 range
                time = (time + ride_duration) % 24 
                # Calculate the number of days
                num_days = (time + ride_duration) // 24
                # convert days mapped between 0 to 6 range
                day = (day + num_days ) % 7
            return time, day
    
    # Define the step function
    def step(self, state, action, Time_matrix):
        """Wraps the funtionality of determining reward, determining next state and 
        total time taken for the step in this step function."""
        reward = self.reward_func(state , action , Time_matrix)
        next_state, step_time = self.next_state_func(state, action, Time_matrix)
        return reward , next_state , step_time

    def reset(self):
        return self.action_space, self.state_space, self.state_init
