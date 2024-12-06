import pybullet as p
import pybullet_data
import time
from useful_code import *
import random
from matplotlib import pyplot as plt
import numpy as np   

def check_node_collision(robot_id, object_ids, joint_position):
    """
    Checks for collisions between a robot and an object in PyBullet. 

    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_id (int): The ID of the object in PyBullet.
        joint_position (list): List of joint positions. 

    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    # set joint positions
    for joint_index, joint_pos in enumerate(joint_position):
        p.resetJointState(robot_id, joint_index, joint_pos)

    # Perform collision check for all links
    for object_id in object_ids:    # Check for each object
        for link_index in range(0, p.getNumJoints(robot_id)): # Check for each link of the robot
            contact_points = p.getClosestPoints(
                bodyA=robot_id, bodyB=object_id, distance=0.01, linkIndexA=link_index
            )
            if contact_points:  # If any contact points exist, a collision is detected
                return True # exit early
    return False

#################################################
#### YOUR CODE HERE: COLLISION EDGE CHECKING ####
#################################################
def check_edge_collision(robot_id, object_ids, joint_position_start, joint_position_end, discretization_step=0.01):
    """ 
    Checks for collision between two joint positions of a robot in PyBullet.
    Args:
        robot_id (int): The ID of the robot in PyBullet.
        object_ids (list): List of IDs of the objects in PyBullet.
        joint_position_start (list): List of joint positions to start from.
        joint_position_end (list): List of joint positions to get to.
        discretization_step (float): maximum interpolation distance before a new collision check is performed.
    Returns:
        bool: True if a collision is detected, False otherwise.
    """
    
    joint_position_start = np.array(joint_position_start)
    joint_position_end = np.array(joint_position_end)
    
    # Compute the total distance in joint space
    total_distance = np.linalg.norm(joint_position_end - joint_position_start)
    
    # Calculate the number of steps based on discretization_step
    num_steps = max(int(np.ceil(total_distance / discretization_step)), 1)
    
    # Generate interpolated joint positions between start and end positions
    for i in range(num_steps + 1):
        t = i / num_steps  # Interpolation factor between 0 and 1
        intermediate_joint_position = (1 - t) * joint_position_start + t * joint_position_end
        
        # Check for collision at the intermediate joint position
        if check_node_collision(robot_id, object_ids, intermediate_joint_position):
            return True  # Collision detected
    
    # No collision detected along the path
    return False

# Provided 
class Node:
    def __init__(self, joint_angles):
        self.joint_angles = np.array(joint_angles)  # joint angles of the node in n-dimensional space
        self.parent = None
        self.cost = 0.0  # Cost from the start node to this node

######################################################################
##################### YOUR CODE HERE: RRT CLASS ######################
######################################################################
class RRT:
    def __init__(self, q_start, q_goal, robot_id, obstacle_ids, q_limits, max_iter=10000, step_size=0.5, search_radius=1.0):
        """
        RRT Initialization.

        Parameters:
        - q_start: List of starting joint angles [x1, x2, ..., xn].
        - q_goal: List of goal joint angles [x1, x2, ..., xn].
        - obstacle_ids: List of obstacles, each as a tuple ([center1, center2, ..., centern], radius).
        - q_limits: List of tuples [(min_x1, max_x1), ..., (min_xn, max_xn)] representing the limits in each dimension.
        - max_iter: Maximum number of iterations.
        - step_size: Maximum step size to expand the tree.
        - search_radius: Radius to search for nearby nodes for rewiring.
        """
        self.q_start = Node(q_start)
        self.q_goal = Node(q_goal)
        self.obstacle_ids = obstacle_ids
        self.robot_id = robot_id
        self.q_limits = q_limits
        self.max_iter = max_iter
        self.step_size = step_size
        self.node_list = [self.q_start]
        self.search_radius = search_radius

    def step(self, from_node, to_joint_angles):
        """Step from 'from_node' towards 'to_joint_angles'. Returns a new node.

        If the distance between 'from_node' and 'to_joint_angles' is less than self.step_size,
        returns 'to_joint_angles'. Otherwise, returns a new node at a distance of self.step_size
        from 'from_node' towards 'to_joint_angles'.
        """
        from_angles = from_node.joint_angles
        to_angles = np.array(to_joint_angles)
        direction = to_angles - from_angles
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            new_angles = to_angles
        else:
            direction_normalized = direction / distance
            new_angles = from_angles + direction_normalized * self.step_size

        new_node = Node(new_angles)
        new_node.parent = from_node
        new_node.cost = from_node.cost + distance
        return new_node

    def get_nearest_node(self, random_point):
        """Find the nearest node in the tree to a given random_point."""
        random_point = np.array(random_point)
        min_distance = float('inf')
        nearest_node = None
        for node in self.node_list:
            distance = np.linalg.norm(node.joint_angles - random_point)
            if distance < min_distance:
                min_distance = distance
                nearest_node = node
        return nearest_node
    
    def find_nearby_nodes(self, new_node):
        """Find nodes within a certain radius of the new node."""
        n = len(self.node_list)
        r = min(self.search_radius * np.sqrt((np.log(n) / n)), self.step_size)
        nearby_nodes = []
        for node in self.node_list:
            distance = np.linalg.norm(node.joint_angles - new_node.joint_angles)
            if distance <= r:
                nearby_nodes.append(node)
        return nearby_nodes
    
    def plan(self):
        """Run the RRT algorithm to find a path of dimension Nx3. Limit the search to only max_iter iterations."""
        goal_sample_rate = 0.1  # 10% chance to sample the goal directly
        for i in range(self.max_iter):
            # Goal biasing
            if np.random.rand() < goal_sample_rate:
                random_point = self.q_goal.joint_angles.tolist()
            else:
                # Sample random point within joint limits
                random_point = [np.random.uniform(low, high) for (low, high) in self.q_limits]
                # print(f"Iteration {i}: Sampling random point {random_point}")

            # Find the nearest node in the tree
            nearest_node = self.get_nearest_node(random_point)

            # Steer from nearest node towards random point
            new_node = self.step(nearest_node, random_point)

            # Check for collision
            collision_edge = check_edge_collision(
                self.robot_id,
                self.obstacle_ids,
                nearest_node.joint_angles.tolist(),
                new_node.joint_angles.tolist()
            )
            collision_node = check_node_collision(
                self.robot_id,
                self.obstacle_ids,
                new_node.joint_angles.tolist()
            )

            if not collision_edge and not collision_node:
                self.node_list.append(new_node)

                # Check if goal is reached
                if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) <= self.step_size:
                    # Connect to goal node
                    goal_node = Node(self.q_goal.joint_angles.tolist())
                    goal_node.parent = new_node
                    self.node_list.append(goal_node)

                    # Retrieve path
                    path = []
                    node = goal_node
                    while node is not None:
                        path.append(node.joint_angles.tolist())
                        node = node.parent
                    path.reverse()
                    return np.array(path)

        # If max_iter is reached without finding a path
        print("Failed to find a path within the maximum iterations")
        return None
    
    def plan2(self):
        """Run the RRT* algorithm to find an optimized path."""
        goal_sample_rate = 0.1  # 10% chance to sample the goal directly
        for i in range(self.max_iter):
            # Goal biasing
            if np.random.rand() < goal_sample_rate:
                random_point = self.q_goal.joint_angles.tolist()
            else:
                random_point = [np.random.uniform(low, high) for (low, high) in self.q_limits]

            # Find nearest node
            nearest_node = self.get_nearest_node(random_point)

            # Steer towards random point
            new_node = self.step(nearest_node, random_point)

            # Collision checking
            if not check_edge_collision(
                self.robot_id,
                self.obstacle_ids,
                nearest_node.joint_angles.tolist(),
                new_node.joint_angles.tolist()
            ) and not check_node_collision(
                self.robot_id,
                self.obstacle_ids,
                new_node.joint_angles.tolist()
            ):
                # Find nearby nodes for rewiring
                nearby_nodes = self.find_nearby_nodes(new_node)

                # Initialize cost and parent
                min_cost = nearest_node.cost + np.linalg.norm(nearest_node.joint_angles - new_node.joint_angles)
                min_node = nearest_node

                # Choose the parent node that results in the lowest cost
                for node in nearby_nodes:
                    if not check_edge_collision(
                        self.robot_id,
                        self.obstacle_ids,
                        node.joint_angles.tolist(),
                        new_node.joint_angles.tolist()
                    ):
                        cost = node.cost + np.linalg.norm(node.joint_angles - new_node.joint_angles)
                        if cost < min_cost:
                            min_cost = cost
                            min_node = node

                # Set the parent and cost of new_node
                new_node.parent = min_node
                new_node.cost = min_cost
                self.node_list.append(new_node)

                # Rewire the tree
                for node in nearby_nodes:
                    if node == min_node:
                        continue
                    if not check_edge_collision(
                        self.robot_id,
                        self.obstacle_ids,
                        new_node.joint_angles.tolist(),
                        node.joint_angles.tolist()
                    ):
                        new_cost = new_node.cost + np.linalg.norm(new_node.joint_angles - node.joint_angles)
                        if new_cost < node.cost:
                            node.parent = new_node
                            node.cost = new_cost

                # Check if goal is reached
                if np.linalg.norm(new_node.joint_angles - self.q_goal.joint_angles) <= self.step_size:
                    # Connect to goal node
                    goal_node = Node(self.q_goal.joint_angles.tolist())
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + np.linalg.norm(new_node.joint_angles - goal_node.joint_angles)
                    self.node_list.append(goal_node)

                    # Retrieve path
                    path = []
                    node = goal_node
                    while node is not None:
                        path.append(node.joint_angles.tolist())
                        node = node.parent
                    path.reverse()
                    return np.array(path)

        # If max_iter is reached without finding a path
        print("Failed to find a path within the maximum iterations")
        return None

#####################################################
##################### MAIN CODE #####################
#####################################################

if __name__ == "__main__":
    
    #######################
    #### PROBLEM SETUP ####
    #######################

    # Initialize PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # For default URDFs
    p.setGravity(0, 0, -9.8)

    # Load the plane and robot arm
    ground_id = p.loadURDF("plane.urdf")
    arm_id = p.loadURDF("three_link_arm.urdf", [0, 0, 0], useFixedBase=True)

    # Add Collision Objects
    collision_ids = [ground_id] # add the ground to the collision list
    collision_positions = [[0.3, 0.5, 0.251], [-0.3, 0.3, 0.101], [-1, -0.15, 0.251], [-1, -0.15, 0.752], [-0.5, -1, 0.251], [0.5, -0.35, 0.201], [0.5, -0.35, 0.602]]
    collision_orientations =  [[0, 0, 0.5], [0, 0, 0.2], [0, 0, 0],[0, 0, 1], [0, 0, 0], [0, 0, .25], [0, 0, 0.5]]
    collision_scales = [0.5, 0.25, 0.5, 0.5, 0.5, 0.4, 0.4]
    for i in range(len(collision_scales)):
        collision_ids.append(p.loadURDF("cube.urdf",
            basePosition=collision_positions[i],  # Position of the cube
            baseOrientation=p.getQuaternionFromEuler(collision_orientations[i]),  # Orientation of the cube
            globalScaling=collision_scales[i]  # Scale the cube to half size
        ))

    # Goal Joint Positions for the Robot
    goal_positions = [[-2.54, 0.15, -0.15], [-1.82,0.15,-0.15],[0.5, 0.15,-0.15], [1.7,0.2,-0.15],[-2.54, 0.15, -0.15]]

    # Joint Limits of the Robot
    joint_limits = [[-np.pi, np.pi], [0, np.pi], [-np.pi, np.pi]]

    # A3xN path array that will be filled with waypoints through all the goal positions
    path_saved = [goal_positions[0]] # Start at the first goal position

    ################
    #### Part 3 ####
    ################

    ##### RUN RRT MOTION PLANNER FOR ALL goal_positions (starting at goal position 1) #####

    print("Begin path planning.")

    # Plan paths between all goal positions
    for i in range(len(goal_positions) - 1):
        q_start = goal_positions[i]
        q_goal = goal_positions[i + 1]

        # Create an instance of the RRT planner
        rrt_planner = RRT(
            q_start=q_start,
            q_goal=q_goal,
            robot_id=arm_id,
            obstacle_ids=collision_ids,
            q_limits=joint_limits,
            max_iter=5000,
            step_size=0.1
        )

        # Run the RRT planner
        path = rrt_planner.plan()

        if path is not None:
            # Append the planned path to path_saved, excluding the first point to avoid duplication
            for joint_angles in path[1:]:
                path_saved.append(joint_angles.tolist())
            print(f"Path from position {i} to {i + 1} found.")
        else:
            print(f"No path found from position {i} to {i + 1}.")

    # Convert path_saved to a NumPy array
    path_saved = np.array(path_saved)

    ###### RUN THE SIMULATION AND MOVE THE ROBOT ALONG PATH_SAVED #####

    print("Start simulation")

    # Set the initial joint positions
    for joint_index, joint_pos in enumerate(goal_positions[0]):
        p.resetJointState(arm_id, joint_index, joint_pos)

    # Move through the waypoints
    for waypoint in path_saved:
        # Move to next waypoint
        for joint_index, joint_pos in enumerate(waypoint):
            # Run position control to reach the waypoint
            p.setJointMotorControl2(
                bodyIndex=arm_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_pos,
                force=500,
                positionGain=0.03,
                velocityGain=1
            )
        # Simulate for a short duration to allow movement
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    print("Simulation end.")

    ################
    #### Bonus ####
    ################

    ##### RUN RRT* MOTION PLANNER FOR ALL goal_positions (starting at goal position 1) #####

    path_saved = [goal_positions[0]]

    print("Begin path planning.")

    # Plan paths between all goal positions
    for i in range(len(goal_positions) - 1):
        q_start = goal_positions[i]
        q_goal = goal_positions[i + 1]

        # Create an instance of the RRT planner
        rrt_planner = RRT(
            q_start=q_start,
            q_goal=q_goal,
            robot_id=arm_id,
            obstacle_ids=collision_ids,
            q_limits=joint_limits,
            max_iter=5000,
            step_size=0.1,
            search_radius=1.0
        )

        # Run the RRT* planner
        path = rrt_planner.plan2()

        if path is not None:
            # Append the planned path to path_saved, excluding the first point to avoid duplication
            for joint_angles in path[1:]:
                path_saved.append(joint_angles.tolist())
            print(f"Path from position {i} to {i + 1} found.")
        else:
            print(f"No path found from position {i} to {i + 1}.")

    # Convert path_saved to a NumPy array
    path_saved = np.array(path_saved)

    ###### RUN THE SIMULATION AND MOVE THE ROBOT ALONG PATH_SAVED #####

    print("Start simulation")

    # Set the initial joint positions
    for joint_index, joint_pos in enumerate(goal_positions[0]):
        p.resetJointState(arm_id, joint_index, joint_pos)

    # Move through the waypoints
    for waypoint in path_saved:
        # Move to next waypoint
        for joint_index, joint_pos in enumerate(waypoint):
            # Run position control to reach the waypoint
            p.setJointMotorControl2(
                bodyIndex=arm_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=joint_pos,
                force=500,
                positionGain=0.03,
                velocityGain=1
            )
        # Simulate for a short duration to allow movement
        for _ in range(50):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    print("Simulation end.")

    # Disconnect from PyBullet
    time.sleep(100) # Remove this line -- it is just to keep the GUI open when you first run this starter code
    p.disconnect()