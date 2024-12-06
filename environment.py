import pybullet as p
import pybullet_data

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For default URDFs
p.setGravity(0, 0, -9.8)

# Load the plane
plane_id = p.loadURDF("plane.urdf")

# Load the Panda robot arm
panda_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# Add a single cube object
cube_position = [0.5, 0, 0.1]  # Position in [x, y, z]
cube_orientation = p.getQuaternionFromEuler([0, 0, 0])  # No rotation
cube_id = p.loadURDF("cube.urdf", basePosition=cube_position, baseOrientation=cube_orientation, globalScaling=0.2)

# Keep simulation running
while True:
    p.stepSimulation()
