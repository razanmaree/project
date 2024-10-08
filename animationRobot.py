import vedo as vd
vd.settings.default_backend= 'vtk'
import numpy as np
from manim import *
import time



#sound
import platform

if platform.system() == "Windows":
    import winsound
else:
    import os

def play_sound(sound_file):
    if platform.system() == "Windows":
        winsound.PlaySound(sound_file, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        if platform.system() == "Darwin":  # macOS
            os.system(f"afplay {sound_file} &")
        else:  # Linux and other Unix-like systems
            os.system(f"aplay {sound_file} &")




num_links = 5
link_length = 1 
LIGHT_GRAY = rgb_to_color([210,210,210])  
DARK_GRAY = rgb_to_color([140,160,160])  
RED = rgb_to_color([1, 0, 0])  
IK_target = [1, 1, 0]


def Rot(angle, axis):

    axis = np.array(axis)
    axis = axis/np.linalg.norm(axis)
    I = np.eye(3)
    K = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = I + np.sin(angle)*K + (1-np.cos(angle))*np.dot(K,K)
    return R
    
class SimpleArm:
    def __init__(self, n=num_links, link_lengths=None, offset=np.array([0, 0, 0]),  alignment_axis='x', link_colors=None):
        
        self.n = n # number of links
        self.angles = [0]*self.n # joint angles, starting from the base joint to the end effector

        self.initial_offset = offset  # Store the initial offset
        self.offset = np.array([0, 0, 0])  # Default offset to zero (this will be updated in FK only initially)

        if link_lengths is None:
            self.link_lengths = [link_length] * self.n  # Default to uniform length if not specified
        else:
            assert len(link_lengths) == n, "Number of link lengths must match number of links"
            self.link_lengths = link_lengths

        self.link_colors = link_colors if link_colors is not None else [LIGHT_GRAY] * self.n
 
        self.offset = offset  
        self.alignment_axis = alignment_axis

        self.child_arms = []  # List to store child arms
        self.parent_arm = None  # Reference to parent arm if this is a child arm
        self.parent_joint = None  # Reference to parent joint if this is a child arm
        self.child_connection_point = 0  # Which joint of this arm connects to the psarent

        # self.Jl is a matrix that contains joints position in local coordinates.
        # Each row contains the coordinates of a joint
        # Number of joints is n+1, including the base and end effector
        self.Jl = np.zeros((self.n+1, 3)) 
    
        self.initialize_joint_positions()

        self.Jw = np.zeros((self.n+1, 3)) # joint positions in world coordinates

        self.joint_shapes = []  # List to hold multiple shapes 
        self.link_shapes = []  # list to hold link shapes

        self.FK()

        #initial shape position
        self.update_shape_positions()  


    def initialize_joint_positions(self):
        if self.alignment_axis == 'y':
            for i in range(1, self.n + 1):
                self.Jl[i, :] = np.array([0, self.link_lengths[i-1], 0])
        else:  # default to x-axis
            for i in range(1, self.n + 1):
                self.Jl[i, :] = np.array([self.link_lengths[i-1], 0, 0])


    def add_joint_shape(self, shape, joint_index, initial_rotation=0, should_rotate=True, offset=(0, 0, 0)):
        self.joint_shapes.append((shape, joint_index, initial_rotation, should_rotate, offset))


    def add_link_shape(self, shape, link_index, initial_rotation=0):
        self.link_shapes.append((shape, link_index, initial_rotation))


    def update_shape_positions(self):
        for shape, joint_index, initial_rotation, should_rotate, offset in self.joint_shapes:
            joint_position = self.Jw[joint_index]
            offset_position = joint_position + np.array(offset)
            shape.pos(offset_position)  # Update position
            
            # Only rotate the shape if should_rotate is True and relative to the joint's rotation
            if should_rotate:
                if joint_index < len(self.angles):
                    # Rotate around the link/joint axis
                    shape.rotate(self.angles[joint_index] + initial_rotation, axis=[0, 0, 1], point=joint_position)
            
        for shape, link_index, initial_rotation in self.link_shapes:
            start_pos = self.Jw[link_index]
            end_pos = self.Jw[link_index + 1]
            mid_pos = (start_pos + end_pos) / 2
            shape.pos(mid_pos)  # Update position
            
            direction = end_pos - start_pos
            angle = np.arctan2(direction[1], direction[0])  # Rotation along the link
            shape.rotate(angle + initial_rotation, axis=[0, 0, 1], point=mid_pos)



    def FK(self, angles=None, apply_initial_offset=True):
        if angles is not None:
            self.angles = angles

        # Only apply the initial offset during the first FK call
        if apply_initial_offset:
            self.offset = self.initial_offset

        Ri = np.eye(3)
        self.Jw[0, :] = self.offset  # Apply offset to the base joint

        for i in range(1, self.n + 1):
            Ri = Rot(self.angles[i - 1], [0, 0, 1]) @ Ri
            self.Jw[i, :] = Ri @ self.Jl[i, :] + self.Jw[i - 1, :]

        self.update_shape_positions()  # Update the shape positions along with links and joints

        #Update child arms
        for child_arm in self.child_arms:
            #Ensure the child's offset is correctly applied based on the parent joint
            child_arm.offset = self.Jw[child_arm.parent_joint] + child_arm.initial_offset
            child_arm.FK(apply_initial_offset=False)  # Apply the FK to the child arms recursively

        return self.Jw[-1, :] 




    def inverse_kinematics_Gauss_Newton(self, target_position, tolerance=0.01, max_iterations=100, damping=0.01):

        target_position = np.array(target_position)
        
        for iteration in range(max_iterations):
            end_effector_position = self.Jw[-1]  # Current end-effector position
            
            # Check if the end effector is within the tolerance
            error = target_position - end_effector_position
            if np.linalg.norm(error) < tolerance:
                print(f"Reached target in {iteration} iterations!")
                return
            
            # Compute the Jacobian matrix
            J = self.compute_jacobian()
            
            # Gauss-Newton step (with Levenberg-Marquardt damping)
            JTJ = J.T @ J
            delta_theta = np.linalg.solve(JTJ + damping * np.eye(JTJ.shape[0]), J.T @ error)
            
            # Update joint angles
            self.angles += delta_theta
            
            # Update forward kinematics
            self.FK(apply_initial_offset=False)
        
        print(f"Reached max iterations without fully reaching target. Final error: {np.linalg.norm(target_position - self.Jw[-1])}")


    def inverse_kinematics_Gradient_Descent(self, target_position, learning_rate=0.01, tolerance=0.01, max_iterations=1000):

        target_position = np.array(target_position)
        
        for iteration in range(max_iterations):
            end_effector_position = self.Jw[-1]  # Current end-effector position
            
            # Check if the end effector is within the tolerance
            error = target_position - end_effector_position
            if np.linalg.norm(error) < tolerance:
                print(f"Reached target in {iteration} iterations!")
                return
            
            # Compute the Jacobian matrix
            J = self.compute_jacobian()
            
            # Compute the gradient
            gradient = J.T @ error
            
            # Update joint angles using gradient descent
            self.angles += learning_rate * gradient
            
            # Update forward kinematics
            self.FK(apply_initial_offset=False)
        
        print(f"Reached max iterations without fully reaching target. Final error: {np.linalg.norm(target_position - self.Jw[-1])}")


    def compute_jacobian(self):

        J = np.zeros((3, self.n))
        
        for i in range(self.n):
            # Compute the axis of rotation for this joint
            if i == 0:
                axis = np.array([0, 0, 1])  # Assuming rotation around z-axis for the base joint
            else:
                axis = np.array([0, 0, 1])  # For 2D case, all joints rotate around z-axis
            
            # Compute the vector from this joint to the end effector
            r = self.Jw[-1] - self.Jw[i]
            
            # Compute the column of the Jacobian
            J[:, i] = np.cross(axis, r)
        
        return J


    def draw(self):
        vd_arm = vd.Assembly()
        
        # Draw base joint
        base_joint = self.create_joint_rectangle(self.Jw[0,:], 0)
        vd_arm += base_joint
        
        for i in range(1, self.n+1):
            # Draw link
            start, end = self.Jw[i-1,:], self.Jw[i,:]
            link_rectangle = self.create_link_rectangle(start, end, i-1)  # Pass i-1 as link index
            vd_arm += link_rectangle
            
            # Draw joint (except for the end effector)
            if i < self.n:
                joint_rectangle = self.create_joint_rectangle(self.Jw[i,:], i)
                vd_arm += joint_rectangle

        # Add custom shapes
        for shape, _, _, _, _ in self.joint_shapes:  # Updated to unpack 5 values
            vd_arm += shape
        for shape, _, _ in self.link_shapes:
            vd_arm += shape
        
        # Draw child arms
        for child_arm in self.child_arms:
            vd_arm += child_arm.draw()

        return vd_arm
    

    def create_link_rectangle(self, start, end, link_index):
        link_vector = end - start
        link_length = np.linalg.norm(link_vector)
        link_angle = np.arctan2(link_vector[1], link_vector[0])
        
        link_width = 0.3  
        manim_rectangle = Rectangle(width=link_length, height=link_width, color=self.link_colors[link_index])
        
        manim_rectangle.rotate(link_angle)
        manim_rectangle.move_to((start + end) / 2)
        
        return manim_to_vedo(manim_rectangle)

    def create_joint_rectangle(self, position, joint_index):
        joint_width = 0.3  
        joint_height = 0.3  
        manim_rectangle = Rectangle(width=joint_width, height=joint_height, color=DARK_GRAY)
        
        manim_rectangle.move_to(position)
        if joint_index < len(self.angles):
            manim_rectangle.rotate(self.angles[joint_index])
        
        return manim_to_vedo(manim_rectangle)


    def add_child_arm(self, child_arm, parent_joint_index, child_connection_point=0):
        child_arm.parent_arm = self
        child_arm.parent_joint = parent_joint_index
        child_arm.child_connection_point = child_connection_point
        self.child_arms.append(child_arm)   




arms = [
    SimpleArm(n=4, link_lengths=[1, 1, 1, 0.5], offset=np.array([0, 0, 0]), alignment_axis='y')
]



child_arm0 = SimpleArm(n=2, link_lengths=[1, 1], offset=np.array([-1, 0, 0]), alignment_axis='x' ,       
                                                      link_colors=[LIGHT_GRAY, LIGHT_GRAY])  #left hand
child_arm1 = SimpleArm(n=2, link_lengths=[1, 1], offset=np.array([1, 0, 0]), alignment_axis='x' ,          
                                                      link_colors=[LIGHT_GRAY, LIGHT_GRAY])  #right hand


child_arm2 = SimpleArm(n=2, link_lengths=[1, 1], offset=np.array([0.7, -0.75, 0]), alignment_axis='y', 
                                                      link_colors=[LIGHT_GRAY, LIGHT_GRAY])  #right leg
child_arm3 = SimpleArm(n=2, link_lengths=[1, 1], offset=np.array([-0.7, -0.75, 0]), alignment_axis='y',   
                                                      link_colors=[LIGHT_GRAY, LIGHT_GRAY])  #left leg

arms[0].add_child_arm(child_arm0, 2,3)
arms[0].add_child_arm(child_arm1, 2,3)
arms[0].add_child_arm(child_arm2, 0,2)
arms[0].add_child_arm(child_arm3, 0,2)


# Rotate child_arm0 by 180 degrees
child_arm0.angles[0] = np.radians(180)
child_arm0.FK()  # Update the forward kinematics to apply the rotation


child_arm2.angles[0] = np.radians(180)
child_arm2.FK() 

child_arm3.angles[0] = np.radians(180)
child_arm3.FK()  



def create_manim_rectangle(width, height, color=GRAY):
    rectangle = Rectangle(width=width, height=height, color=color)
    return rectangle

def create_manim_rounded_rectangle(width, height, corner_radius=0.1, color=GRAY):
    corner_radius = float(corner_radius)  
    rounded_rectangle = RoundedRectangle(width=width, height=height, corner_radius=corner_radius, color=color)
    return rounded_rectangle

def create_manim_circle(radius, color=GRAY):
    circle = Circle(radius=radius, color=color)
    return circle

def create_manim_half_circle(width, color=GRAY, side='right'):
    radius = width / 2
    circle = Circle(radius=radius, color=color, fill_opacity=1)
    if side == 'right':
        circle.shift(radius * LEFT)  # Move left to create right half
    elif side == 'left':
        circle.shift(radius * RIGHT)  # Move right to create left half
    elif side == 'down':
        circle.shift(radius * UP)  # Move up to create bottom half       
    return circle

def create_manim_arc(width, angle, color=GRAY, stroke_width=4):
    start = np.array([-width/2, 0, 0])
    end = np.array([width/2, 0, 0])
    arc = ArcBetweenPoints(start, end, angle=angle, color=color, stroke_width=stroke_width)
    return arc


def create_manim_downward_arc(width, color=GRAY, stroke_width=2):
    start = np.array([-width/2, 0, 0])
    end = np.array([width/2, 0, 0])
    arc = ArcBetweenPoints(start, end, angle=-PI/6, color=color, stroke_width=stroke_width)
    return arc

def create_manim_block_arc(width, height, angle=PI/2, color=GRAY, joint_index=None):
    # Create the outer arc
    outer_arc = Arc(radius=width/2, angle=angle, color=color, fill_opacity=1)
    # Create the inner arc (to create the hollow effect)
    inner_arc = Arc(radius=width/2 - height, angle=angle, color=BLACK, fill_opacity=1)
    # Use always_redraw to ensure the inner arc stays aligned with the outer arc
    inner_arc.add_updater(lambda m: m.move_arc_center_to(outer_arc.get_arc_center()))
    # Group the arcs together
    block_arc = VGroup(outer_arc, inner_arc)
    block_arc.joint_index = joint_index  # Store the joint index
    return block_arc

def create_manim_broken_line(width, height, num_segments=5, color=BLACK):
    broken_line = VGroup()
    segment_width = width / num_segments
    for i in range(num_segments):
        if i % 2 == 0:
            start = np.array([i * segment_width - width/2, 0, 0])
            end = np.array([(i+1) * segment_width - width/2, 0, 0])
            segment = Line(start=start, end=end, color=color, stroke_width=4)
            broken_line.add(segment)
    return broken_line

def manim_color_to_rgb(manim_color):
    hex_color = manim_color.to_hex()
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def manim_to_vedo(manim_object):
    if isinstance(manim_object, Circle):
        if manim_object.fill_opacity > 0:
            vd_circle = vd.Circle(r=manim_object.radius, c=manim_color_to_rgb(manim_object.color), res=50)
            center = manim_object.get_center()
            if center[0] < 0:  # Left half
                vd_circle.cut_with_plane(normal=(1,0,0), origin=(0,0,0))
            elif center[0] > 0:  # Right half
                vd_circle.cut_with_plane(normal=(-1,0,0), origin=(0,0,0))
            elif center[1] > 0:  # Bottom half
                vd_circle.cut_with_plane(normal=(0,-1,0), origin=(0,0,0))
            return vd_circle
        else:
            return vd.Circle(r=manim_object.radius, c=manim_color_to_rgb(manim_object.color), res=50)
    elif isinstance(manim_object, (Rectangle, RoundedRectangle)):
        points = np.array([v for v in manim_object.get_vertices()])
        faces = np.array([list(range(len(points)))])
        mesh = vd.Mesh([points, faces])
        mesh.c(manim_color_to_rgb(manim_object.color))
        return mesh
    elif isinstance(manim_object, (Arc, ArcBetweenPoints)):
        if isinstance(manim_object, Arc) and manim_object.fill_opacity > 0:
            # For filled half-circle
            vd_circle = vd.Circle(r=manim_object.radius, c=manim_color_to_rgb(manim_object.color), res=50)
            vd_circle.cut_with_plane(normal=(0,-1,0), origin=(0,0,0))  # Cut the lower half
            return vd_circle
        else:
            # For curved line (including downward arc)
            points = np.array([manim_object.point_from_proportion(t) for t in np.linspace(0, 1, 50)])
            vd_line = vd.Line(points, c=manim_color_to_rgb(manim_object.color), lw=manim_object.stroke_width)
            return vd_line
    elif isinstance(manim_object, VGroup):
        if all(isinstance(obj, Line) for obj in manim_object):
            # This is our broken line
            points = []
            for line in manim_object:
                points.extend([line.start[:2], line.end[:2]])  # Only take x and y coordinates
            vd_line = vd.Line(points, c=manim_color_to_rgb(manim_object[0].color), lw=4)
            return vd_line
        elif hasattr(manim_object, 'joint_index'):  # For block arc
            outer_arc = manim_object[0]
            inner_arc = manim_object[1]
            
            # Determine which half to keep based on the joint index
            joint_index = manim_object.joint_index
            
            # Create the outer arc using Circle
            vd_outer = vd.Circle(r=outer_arc.radius, res=50)
            
            # Create the inner arc using Circle
            vd_inner = vd.Circle(r=inner_arc.radius, res=50)
            
            if joint_index == 6:  # First joint, keep right half
                vd_outer.cut_with_plane(origin=(0,0,0), normal=(-1,0,0))
                vd_inner.cut_with_plane(origin=(0,0,0), normal=(-1,0,0))
            elif joint_index == 0:  # Last joint, keep left half
                vd_outer.cut_with_plane(origin=(0,0,0), normal=(1,0,0))
                vd_inner.cut_with_plane(origin=(0,0,0), normal=(1,0,0))
            
            vd_outer.rotate_z(PI/2)  # Rotate to point upwards
            vd_inner.rotate_z(PI/2)  
            
            vd_outer.c(manim_color_to_rgb(outer_arc.color))
            vd_inner.c(LIGHT_GRAY)  # Light gray for inner arc
            
            # Create an assembly of the two arcs
            vd_block_arc = vd.Assembly([vd_outer, vd_inner])
            
            return vd_block_arc
        else:
            raise ValueError("Unsupported VGroup type")
    elif isinstance(manim_object, Line):
        # Handle single line objects
        points = [manim_object.start[:2], manim_object.end[:2]]
        vd_line = vd.Line(points, c=manim_color_to_rgb(manim_object.color))
        return vd_line
    else:
        raise ValueError("Unsupported shape type")



arm_shapes = [
    [
        (2, 1.5, 0, 'rounded_rectangle', 'joint', DARK_GRAY, 0.1, False),#0
        (1, 0.5, 1, 'rectangle', 'joint', DARK_GRAY, False),#1
        (1.6, 1.1, 4, 'rounded_rectangle', 'joint', DARK_GRAY, 0.1, False),#2
        #Face
        # Eyes
        (1.25, 0.5, 4, 'rectangle', 'joint', LIGHT_GRAY, False, (0, 0.2, 0)),#3 #background of the eyes 
        (0.15, None, 4, 'circle', 'joint', WHITE, False, (0.3, 0.25, 0)),#4  # Left eye
        (0.15, None, 4, 'circle', 'joint', WHITE, False, (-0.3, 0.25, 0)),#5  # Right eye

        (0.05, None, 4, 'circle', 'joint', BLACK, False, (0.3, 0.25, 0)),#6  # Inside the Left eye
        (0.05, None, 4, 'circle', 'joint', BLACK, False, (-0.3, 0.25, 0)),#7  # Inside the Right eye
        #eyebrows
        (0.4, None, 4, 'downward_arc', 'joint', BLACK, False, (0.3, 0.45, 0)),#8  # Left eyebrow
        (0.4, None, 4, 'downward_arc', 'joint', BLACK, False, (-0.3, 0.45, 0)),#9  # Right eyebrow  
        # Nose
        (0.1, 0.1, 4, 'rectangle', 'joint', RED, False, (0, 0, 0)),#10
        # Mouth
        (0.4, None, 4, 'arc', 'joint', BLACK, False, (0, -0.3, 0)),#11
        
        #Addetions
        (0.35, 0.2, 4, 'rectangle', 'joint', LIGHT_GRAY, False, (0, 0.65, 0)),#12 #miniTriangle 
        (0.5, None, 4, 'half_circle', 'joint', LIGHT_GRAY, False, (0.8, 0, 0), 'right'),#13  # Right ear
        (0.5, None, 4, 'half_circle', 'joint', LIGHT_GRAY, False, (-0.8, 0, 0), 'left'),#14   # left ear

        (2, 1.5, 2, 'rounded_rectangle', 'joint', DARK_GRAY, 0.1, False),#15

  
    ]

]

# Create a list of all arms
all_arms = [arms[0],child_arm0, child_arm1, child_arm2, child_arm3]

# Create and assign shapes to arms
for arm, shapes in zip(all_arms, arm_shapes):
    for params in shapes:
        shape_type = params[3]
        index = params[2]
        shape_location = params[4]
        color = params[5] if len(params) > 5 else GRAY
        should_rotate = params[6] if len(params) > 6 else True
        offset = params[7] if len(params) > 7 else (0, 0, 0)  # New offset parameter

        if shape_type == 'rectangle':
            width, height = params[0], params[1]
            manim_shape = create_manim_rectangle(width, height, color=color)
        elif shape_type == 'rounded_rectangle':
            width, height = params[0], params[1]
            corner_radius = float(params[6]) if len(params) > 6 else 0.1  # Ensure corner_radius is a float
            manim_shape = create_manim_rounded_rectangle(width, height, corner_radius, color=color)
        elif shape_type == 'circle':
            radius = params[0]
            manim_shape = create_manim_circle(radius, color=color)
        elif shape_type == 'arc' or shape_type == 'downward_arc':
            width = params[0]
            if shape_type == 'arc':
                manim_shape = create_manim_arc(width, angle=PI/6, color=color)
            else:  # downward_arc
                manim_shape = create_manim_downward_arc(width, color=color)
        elif shape_type == 'half_circle':
            width = params[0]
            side = params[8] if len(params) > 8 else 'right'  # Default to right if not specified
            manim_shape = create_manim_half_circle(width, color=color, side=side)
        elif shape_type == 'block_arc':
            width, height = params[0], params[1]
            manim_shape = create_manim_block_arc(width, height, color=color, joint_index=index)
        else:
            raise ValueError(f"Unsupported shape type: {shape_type}")
        
        vedo_shape = manim_to_vedo(manim_shape)
        
        if shape_location == 'joint':
            arm.add_joint_shape(vedo_shape, index, initial_rotation=0, should_rotate=should_rotate, offset=offset)
        elif shape_location == 'link':
            arm.add_link_shape(vedo_shape, index, initial_rotation=0)




def draw_all_arms():
    vd_assembly = vd.Assembly()
    # Draw the parent arm and all its child arms
    arms[0].FK()  # Ensure joint positions are up to date for the main arm
    arms[0].update_shape_positions()  # Update shape positions for the main arm
    vd_assembly += arms[0].draw()
    
    # Explicitly draw all child arms
    #for child_arm in [child_arm0, child_arm1, child_arm2, child_arm3]:
    for child_arm in [arms[0]]:

        child_arm.FK()  # Ensure joint positions are up to date
        child_arm.update_shape_positions()  # Update shape positions
        vd_assembly += child_arm.draw()
    
    return vd_assembly




def wave(arm,arm0):   

    # Play the laughing sound
    play_sound('sound/hello1.wav')

    n = 2   # this is the number of the joint
    t = 0.2 # this is the sleep time
    original_angle = arm.angles[n-1]  # Store the original angle 

    original_angles = arm0.angles.copy()
    arm0.angles[3] = original_angles[3] -  np.radians(20) 

    arm.angles[n-2] = original_angle + np.radians(60)
    time.sleep(t)  

    for _ in range(3):  # Repeat 3 times
        move_to_point(child_arm1, [ 1.73981932e+00 , 3.81695055e+00 ,-4.11406165e-06],"Gauss-Newton") 
        time.sleep(t)  
        
        move_to_point(child_arm1, [ 2.71033078e+00 , 2.96479414e+00, -4.11406165e-06],"Gauss-Newton") 
        time.sleep(t)  

    # Return HEAD to the original position
    arm0.angles[3] = original_angles[3] -  np.radians(0) 
    arm0.FK()  # Update forward kinematics

    # Return to the original position
    arm.angles[n-1] = arm.angles[n-2] = arm.angles[n-3] = original_angle
    arm.FK()  
    plt.remove("Assembly")
    plt.add(draw_all_arms())  
    plt.render()




def blink(arm, eye_indices):
    # Close eyes
    for i in eye_indices:
        arm.joint_shapes[i][0].scale([1, 0.1, 1])
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()
    time.sleep(0.1)
    
    # Open eyes
    for i in eye_indices:
        arm.joint_shapes[i][0].scale([1, 10, 1])  # Scale back to original size
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()



def change_mouth_shape(arm, shape):
    mouth_index = 11  # Correct index for the mouth
    width = 0.4
    stroke_width = 4  # Increase this value to make the line thicker
    if shape == 'smile':
        new_shape = create_manim_arc(width, angle=PI/6, color=BLACK, stroke_width=stroke_width)
    elif shape == 'frown':
        new_shape = create_manim_arc(width, angle=-PI/2, color=BLACK, stroke_width=stroke_width)
    elif shape == 'half_circle':
        new_shape = create_manim_half_circle(width, color=BLACK, side='down')
    elif shape == 'angry':
        new_shape = create_manim_broken_line(width, 0.05, num_segments=7, color=BLACK)
    
    vedo_shape = manim_to_vedo(new_shape)
    arm.joint_shapes[mouth_index] = (vedo_shape, 4, 0, False, (0, -0.3, 0))
    
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()




def cry(arm):
    def create_tear(position):
        return vd.Sphere(r=0.05, c='lightblue').pos(position)

    def animate_tear_pair(left_tear, right_tear, start_y, end_y, frames):
        for i in range(frames):
            new_y = start_y - (start_y - end_y) * (i / frames)
            left_tear.pos([left_tear.pos()[0], new_y, left_tear.pos()[2]])
            right_tear.pos([right_tear.pos()[0], new_y, right_tear.pos()[2]])
            plt.remove("Assembly")
            plt.add(draw_all_arms())
            plt.render()
            time.sleep(0.01)

    # Store the initial angles for arms and legs
    original_arm0_angles = child_arm0.angles.copy()
    original_arm1_angles = child_arm1.angles.copy()


    # Get the positions of both eyes
    left_eye_position = arm.joint_shapes[4][0].pos()
    right_eye_position = arm.joint_shapes[5][0].pos()
    
    # Create three pairs of tears
    tear_pairs = [
        (create_tear(left_eye_position + np.array([0, -0.1 , 0])),
         create_tear(right_eye_position + np.array([0, -0.1 , 0])))
        for i in range(5)
    ]

    #IK
    # move_to_point(child_arm0, [ -5.91775278e-01 , 3.61574695e+00 ,-4.11406165e-06],"Gradient-Descent") 
    # move_to_point(child_arm1, [ 5.91775278e-01 , 3.61574695e+00, -4.11406165e-06],"Gradient-Descent") 

    # Move both arms simultaneously
    move_arms_simultaneously(
        child_arm0, [ -5.91775278e-01 , 3.61574695e+00 ,-4.11406165e-06],
        child_arm1, [ 5.91775278e-01 , 3.61574695e+00, -4.11406165e-06],
        "Gradient-Descent"
    )

    # Change the mouth to a frown
    change_mouth_shape(arm, 'frown')

    # Play the crying sound here
    play_sound('sound/crying1.wav')

    # Close eyes and change inner eye color to black
    eye_indices = [4, 5]  # Outer eyes
    inner_eye_indices = [6, 7]  # Inner eyes
    for i in eye_indices + inner_eye_indices:
        arm.joint_shapes[i][0].scale([1, 0.1, 1])
    for i in eye_indices:
        arm.joint_shapes[i][0].color('black')
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()

    
    # Mouth position 
    end_y = arm.joint_shapes[9][0].pos()[1] - 0.5  # Lowered the end position
    
    # Animate tears falling
    for left_tear, right_tear in tear_pairs:
        plt.add(left_tear)
        plt.add(right_tear)
        
        start_y = left_tear.pos()[1]  # Both tears start at the same y-position
        animate_tear_pair(left_tear, right_tear, start_y, end_y, 5)
        
        plt.remove(left_tear)
        plt.remove(right_tear)
        
        # Short pause between tear pairs
        time.sleep(0.02)
    
    # Open eyes and change inner eye color back to white
    for i in eye_indices + inner_eye_indices:
        arm.joint_shapes[i][0].scale([1, 10, 1])  # Scale back to original size
    for i in eye_indices:
        arm.joint_shapes[i][0].color('white')
    
    reset_face(arm)

    # Restore the initial angles for arms and legs**
    child_arm0.angles = original_arm0_angles.copy()
    child_arm1.angles = original_arm1_angles.copy()


    # Update forward kinematics to apply the restored angles
    child_arm0.FK()
    child_arm1.FK()


    # Redraw one last time to ensure all tears are gone, eyes are open and color is reset
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()




def angry(arm):
    # Store original positions and angles before changing them
    original_angles = arm.angles.copy()
    child0_original_angles = child_arm0.angles.copy()
    child1_original_angles = child_arm1.angles.copy()

    # # Play the screaming sound
    # play_sound('sound/scream3.wav') 
    
    # Make the eyebrows more angled 
    left_eyebrow_index = 8
    right_eyebrow_index = 9

    arm.joint_shapes[left_eyebrow_index][0].rotate(np.radians(800), axis=(0,0,1), point=arm.joint_shapes[left_eyebrow_index][0].pos())
    arm.joint_shapes[right_eyebrow_index][0].rotate(np.radians(-800), axis=(0,0,1), point=arm.joint_shapes[right_eyebrow_index][0].pos())

    # Play the screaming sound
    play_sound('sound/scream3.wav') 

    # Narrow the eyes 
    for i in range(4, 8):
        arm.joint_shapes[i][0].scale([1, 0.5, 1])

    #IK
    # move_to_point(child_arm0, [ -9.70511457e-01 , 6.68706065e-01 ,-4.11406165e-06],"Gradient-Descent") 
    # move_to_point(child_arm1, [ 9.70511457e-01 , 6.68706065e-01 ,-4.11406165e-06],"Gradient-Descent") 

    # Move both arms simultaneously
    move_arms_simultaneously(
        child_arm0, [ -9.70511457e-01 , 6.68706065e-01 ,-4.11406165e-06],
        child_arm1, [ 9.70511457e-01 , 6.68706065e-01 ,-4.11406165e-06],
        "Gradient-Descent"
    )
    
    # Change mouth to show anger
    change_mouth_shape(arm, 'angry')

    # Redraw to display the angry state
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()

    # Hold the angry expression for 3 seconds
    time.sleep(3)

    # Reset to the original state (face, arms, angles, etc.)
    arm.angles = original_angles
    child_arm0.angles = child0_original_angles
    child_arm1.angles = child1_original_angles

    # Reset face expression and arms
    reset_face(arm)
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()




def laughing(arm):
    # Play the laughing sound
    play_sound('sound/laughing1.wav')

    original_position = arm.joint_shapes[0][0].pos()
    original_eye_scales = [shape[0].scale() for shape in arm.joint_shapes[4:8]]

    # Store original angles for arms and legs
    original_arm0_angles = child_arm0.angles.copy()
    original_arm1_angles = child_arm1.angles.copy()
    original_leg2_angles = child_arm2.angles.copy()
    original_leg3_angles = child_arm3.angles.copy()


    # Squint eyes
    for i in range(4, 8):
            arm.joint_shapes[i][0].scale([original_eye_scales[i-4][0], original_eye_scales[i-4][1] * 0.7, original_eye_scales[i-4][2]])

    for i in range(8, 10):
        # Create a new, more curved eyebrow
        eyebrow_width = 0.4
        raised_eyebrow = create_manim_downward_arc(eyebrow_width, color=BLACK)
        arm.joint_shapes[i] = (manim_to_vedo(raised_eyebrow), 4, 0, False, 
                                (0.3 if i == 8 else -0.3, 0.5, 0))  # Moved up slightly

    for _ in range(8):  # Repeat the animation 8 times

        if _%2==0:
            mouth_width = 0.6
            laughing_mouth = create_manim_half_circle(mouth_width, color=BLACK, side='down')
            arm.joint_shapes[11] = (manim_to_vedo(laughing_mouth), 4, 0, False, (0, -0.3, 0))
        else:
            mouth_width = 0.5
            laughing_mouth = create_manim_half_circle(mouth_width, color=BLACK, side='down')
            arm.joint_shapes[11] = (manim_to_vedo(laughing_mouth), 4, 0, False, (0, -0.3, 0))

        
        # Add hand clapping motion
        child_arm0.angles[0] = np.radians(45 * np.sin(_)-180)
        child_arm0.FK()
        child_arm1.angles[0] = np.radians(-45 * np.sin(_))
        child_arm1.FK()

        # Add leg movement to simulate balancing
        child_arm2.angles[0] = np.radians(10 * np.sin(_)-180)
        child_arm3.angles[0] = np.radians(-10 * np.sin(_)-180)
        child_arm2.FK()
        child_arm3.FK()

        plt.remove("Assembly")
        plt.add(draw_all_arms())
        plt.render()
        time.sleep(0.1)

        # Reset position, eyes, eyebrows, and close mouth
        arm.joint_shapes[0][0].pos(original_position)
        for i in range(4, 8):
            arm.joint_shapes[i][0].scale(original_eye_scales[i-4])

        plt.remove("Assembly")
        plt.add(draw_all_arms())
        plt.render()
        time.sleep(0.1)

    #Restore the initial angles for arms and legs
    child_arm0.angles = original_arm0_angles.copy()
    child_arm1.angles = original_arm1_angles.copy()
    child_arm2.angles = original_leg2_angles.copy()
    child_arm3.angles = original_leg3_angles.copy()


    # Reset face at the end
    reset_face(arm)

    # Redraw one last time to ensure everything is reset
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()



def surprised(arm):
    # # Play a surprised sound
    # play_sound('sound/surprised1.wav') 

    original_eye_scales = [shape[0].scale() for shape in arm.joint_shapes[4:8]]

    # Store the initial angles for arms and legs
    original_arm0_angles = child_arm0.angles.copy()
    original_arm1_angles = child_arm1.angles.copy()


    for i in range(8, 10):
        # Create a new, more curved eyebrow
        eyebrow_width = 0.4
        raised_eyebrow = create_manim_downward_arc(eyebrow_width, color=BLACK)
        arm.joint_shapes[i] = (manim_to_vedo(raised_eyebrow), 4, 0, False, 
                                (0.3 if i == 8 else -0.3, 0.5, 0))  # Moved up slightly
            

    # Widen eyes
    for i in range(4, 8):
        arm.joint_shapes[i][0].scale([original_eye_scales[i-4][0] * 1.3, original_eye_scales[i-4][1] * 1.3, original_eye_scales[i-4][2]])

    # Open mouth wide (create a new shape for surprised mouth)
    surprised_mouth = create_manim_circle(0.15, color=BLACK)  # Adjust size as needed
    arm.joint_shapes[11] = (manim_to_vedo(surprised_mouth), 4, 0, False, (0, -0.3, 0)) 


    # move_to_point(child_arm0, [ -4.37913706e-01 , 3.28435280e+00, -4.11406165e-06],"Gauss-Newton") #"Gradient-Descent"
    # move_to_point(child_arm1,  [ 4.37913706e-01 , 3.28435280e+00, -4.11406165e-06],"Gauss-Newton")

    # Move both arms simultaneously
    move_arms_simultaneously(
        child_arm0, [-4.37913706e-01, 3.28435280e+00, -4.11406165e-06],
        child_arm1, [4.37913706e-01, 3.28435280e+00, -4.11406165e-06],
        "Gradient-Descent"
    )

    # Play a surprised sound
    play_sound('sound/surprised1.wav')

    # Render the changes
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()

    # Hold the expression for a moment
    time.sleep(2)

    # Reset face
    reset_face(arm)

    # Restore the initial angles for arms and legs
    child_arm0.angles = original_arm0_angles.copy()
    child_arm1.angles = original_arm1_angles.copy()

    # Update forward kinematics to apply the restored angles
    child_arm0.FK()
    child_arm1.FK()

    # Redraw one last time to ensure all tears are gone, eyes are open and color is reset
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()




def move_arms_simultaneously(arm1, target1, arm2, target2, currentMode="Gauss-Newton", iterations=1000):
    target1 = np.array(target1)
    target2 = np.array(target2)
    
    for iteration in range(iterations):
        # Update arm1
        if currentMode == "Gradient-Descent":
            arm1.inverse_kinematics_Gradient_Descent(target1)
        elif currentMode == "Gauss-Newton":
            arm1.inverse_kinematics_Gauss_Newton(target1, max_iterations=1)  # Only one iteration at a time
            
        # Update arm2
        if currentMode == "Gradient-Descent":
            arm2.inverse_kinematics_Gradient_Descent(target2)
        elif currentMode == "Gauss-Newton":
            arm2.inverse_kinematics_Gauss_Newton(target2, max_iterations=1)  # Only one iteration at a time
            
        # Update the entire robot
        plt.remove("Assembly")
        plt.add(draw_all_arms())
        plt.render()
        
        # Check if both arms are close enough to their targets
        distance1 = np.linalg.norm(arm1.Jw[-1] - target1)
        distance2 = np.linalg.norm(arm2.Jw[-1] - target2)
        if distance1 < 1e-2 and distance2 < 1e-2:
            break



def yawning(arm):
    # Play a yawning sound
    play_sound('sound/yawning2.wav')  

    original_eye_scales = [shape[0].scale() for shape in arm.joint_shapes[4:8]]

    # Store the initial angles for arms and legs
    original_arm0_angles = child_arm0.angles.copy()
    original_arm1_angles = child_arm1.angles.copy()

    #IK
    move_to_point(child_arm1, [ 4.37913706e-01 , 3.28435280e+00, -4.11406165e-06],"Gradient-Descent")  


    # Yawning animation
    for i in range(9):  # 10 frames of animation
        progress = i / 9  # Animation progress from 0 to 1

        # Gradually open mouth
        mouth_size = 0.1 + 0.1 * progress  # Starts at 0.1, ends at 0.5
        yawning_mouth = create_manim_circle(mouth_size, color=BLACK)
        arm.joint_shapes[11] = (manim_to_vedo(yawning_mouth), 4, 0, False, (0, -0.3, 0))

        # Gradually close eyes
        eye_scale = 1 - 0.7 * progress  # Starts at 1, ends at 0.3
        for j in range(4, 8):
            arm.joint_shapes[j][0].scale([original_eye_scales[j-4][0], original_eye_scales[j-4][1] * eye_scale, original_eye_scales[j-4][2]])


        # Render the changes
        plt.remove("Assembly")                                          
        plt.add(draw_all_arms())                                        
        plt.render()
        time.sleep(0.1)

    # Hold the peak yawn position
    time.sleep(1)

    play_sound('sound/yawning1.wav')  

    # Gradually return to normal
    for i in range(9):
        progress = 1 - i / 9  # Animation progress from 1 to 0

        # Reverse the animations
        mouth_size = 0.1 + 0.1 * progress
        yawning_mouth = create_manim_circle(mouth_size, color=BLACK)
        arm.joint_shapes[11] = (manim_to_vedo(yawning_mouth), 4, 0, False, (0, -0.3, 0))

        eye_scale = 1 + 0.99 * progress
        for j in range(4, 8):
            arm.joint_shapes[j][0].scale([original_eye_scales[j-4][0], original_eye_scales[j-4][1] * eye_scale, original_eye_scales[j-4][2]])

        plt.remove("Assembly")                         
        plt.add(draw_all_arms())        
        plt.render()
        time.sleep(0.1)


    # Restore the initial angles for arms and legs
    child_arm0.angles = original_arm0_angles.copy()
    child_arm1.angles = original_arm1_angles.copy()

    # Update forward kinematics to apply the restored angles
    child_arm0.FK()
    child_arm1.FK()

    # Reset face
    reset_face(arm)




def reset_face(arm):
    # Reset eyebrows (indices 8 and 9)
    eyebrow_width = 0.4
    left_eyebrow = create_manim_downward_arc(eyebrow_width, color=BLACK)
    right_eyebrow = create_manim_downward_arc(eyebrow_width, color=BLACK)
    arm.joint_shapes[8] = (manim_to_vedo(left_eyebrow), 4, 0, False, (0.3, 0.45, 0))
    arm.joint_shapes[9] = (manim_to_vedo(right_eyebrow), 4, 0, False, (-0.3, 0.45, 0))
    
    # Reset eyes (indices 4, 5, 6, 7)
    eye_background = create_manim_rectangle(1.25, 0.5, color=LIGHT_GRAY)
    left_eye_outer = create_manim_circle(0.15, color=WHITE)
    right_eye_outer = create_manim_circle(0.15, color=WHITE)
    left_eye_inner = create_manim_circle(0.05, color=BLACK)
    right_eye_inner = create_manim_circle(0.05, color=BLACK)
    
    arm.joint_shapes[3] = (manim_to_vedo(eye_background), 4, 0, False, (0, 0.2, 0))
    arm.joint_shapes[4] = (manim_to_vedo(left_eye_outer), 4, 0, False, (0.3, 0.25, 0))
    arm.joint_shapes[5] = (manim_to_vedo(right_eye_outer), 4, 0, False, (-0.3, 0.25, 0))
    arm.joint_shapes[6] = (manim_to_vedo(left_eye_inner), 4, 0, False, (0.3, 0.25, 0))
    arm.joint_shapes[7] = (manim_to_vedo(right_eye_inner), 4, 0, False, (-0.3, 0.25, 0))
    
    # Reset nose (index 10)
    nose = create_manim_rectangle(0.1, 0.1, color=RED)
    arm.joint_shapes[10] = (manim_to_vedo(nose), 4, 0, False, (0, 0, 0))
    
    # Reset mouth to smile (index 11)
    mouth_width = 0.4
    smile = create_manim_arc(mouth_width, angle=PI/6, color=BLACK, stroke_width=4)
    arm.joint_shapes[11] = (manim_to_vedo(smile), 4, 0, False, (0, -0.3, 0))
    
    # Reset additional features (mini triangle, ears)
    mini_triangle = create_manim_rectangle(0.35, 0.2, color=LIGHT_GRAY)
    right_ear = create_manim_half_circle(0.5, color=LIGHT_GRAY, side='right')
    left_ear = create_manim_half_circle(0.5, color=LIGHT_GRAY, side='left')
    
    arm.joint_shapes[12] = (manim_to_vedo(mini_triangle), 4, 0, False, (0, 0.65, 0))
    arm.joint_shapes[13] = (manim_to_vedo(right_ear), 4, 0, False, (0.8, 0, 0))
    arm.joint_shapes[14] = (manim_to_vedo(left_ear), 4, 0, False, (-0.8, 0, 0))
    
    # Redraw
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()



def shake_head(arm, angle_right, angle_left):

    n = 3  # this is the number of the joint we want to move
    t = 0.2  # this is the sleep time
    original_angles = arm.angles.copy()  # Store all original angles

    # Tilt head to the right
    arm.angles[n] = original_angles[n] - angle_right
    arm.FK()
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()
    time.sleep(t)

    # Tilt head to the left
    arm.angles[n] = original_angles[n] + angle_left
    arm.FK()
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()
    time.sleep(t)


    # Return to the original position
    arm.angles = original_angles
    arm.FK()
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()



def jump(arm, height=1.0, duration=1.0, n=4):      

    # Store the initial offsets and angles
    initial_offset = arm.offset.copy()  # Store the initial offset of the body
    initial_leg1_offset = child_arm2.offset.copy()  
    initial_leg2_offset = child_arm3.offset.copy()  
    initial_child1_offset = child_arm1.offset.copy()  
    initial_child0_offset = child_arm0.offset.copy() 

    # Store the initial angles of the legs and child arms
    initial_leg1_angles = child_arm2.angles.copy()  
    initial_leg2_angles = child_arm3.angles.copy()  
    initial_child1_angles = child_arm1.angles.copy()  
    initial_child0_angles = child_arm0.angles.copy()  

    # Store the initial states of the face components
    original_eyebrows = [arm.joint_shapes[8][0].clone(), arm.joint_shapes[9][0].clone()]  # Eyebrows

    # Calculate the number of frames and the time per frame for smooth animation
    frames = 30
    time_per_frame = duration / frames

    def update_jump_position(t):
        # Use a smooth parabolic equation for the jump: y = -4 * (x - 0.5)^2 + 1, where x goes from 0 to 1
        x = t / duration
        if x > 1.0:
            x = 1.0
        y_offset = -4 * (x - 0.5) ** 2 + 1  # Creates a parabolic motion

        # Update the vertical position of the body and legs
        arm.offset[1] = initial_offset[1] + height * y_offset
        child_arm2.offset[0] = initial_leg1_offset[0] + height * y_offset
        child_arm3.offset[0] = initial_leg2_offset[0] + height * y_offset
        child_arm1.offset[0] = initial_child1_offset[0] + height * y_offset
        child_arm0.offset[0] = initial_child0_offset[0] + height * y_offset

        #Modify facial expressions based on jump height
        if y_offset > 0.5:
            # Raise eyebrows
            eyebrow_width = 0.4
            raised_eyebrow_left = create_manim_downward_arc(eyebrow_width, color=BLACK)
            raised_eyebrow_right = create_manim_downward_arc(eyebrow_width, color=BLACK)
            arm.joint_shapes[8] = (manim_to_vedo(raised_eyebrow_left), 4, 0, False, (0.3, 0.5, 0))  # Left eyebrow
            arm.joint_shapes[9] = (manim_to_vedo(raised_eyebrow_right), 4, 0, False, (-0.3, 0.5, 0))  # Right eyebrow


            mouth_width = 0.6
            laughing_mouth = create_manim_half_circle(mouth_width, color=BLACK, side='down')
            arm.joint_shapes[11] = (manim_to_vedo(laughing_mouth), 4, 0, False, (0, -0.3, 0))


            # Bend legs inward
            child_arm2.angles[0] = np.radians(210)  # Bend the left leg inward
            child_arm3.angles[0] = np.radians(150)  # Bend the right leg inward
            child_arm2.FK()  # Update the forward kinematics for the left leg
            child_arm3.FK()  # Update the forward kinematics for the right leg

            # Rotate arms upward
            child_arm0.angles[0] = np.radians(135)
            child_arm0.FK()  # Update the forward kinematics to apply the rotation
            #move_to_point(child_arm0, [ -2.53811415e+00 , 3.45138380e+00, -4.11406165e-06],"Gauss-Newton") 

            child_arm1.angles[0] = np.radians(45)
            child_arm1.FK()  # Update the forward kinematics to apply the rotation

            #move_to_point(child_arm1, [ 2.59197572e+00 , 4.24884628e+00, -4.11406165e-06],"Gauss-Newton") 
        else:
            # Reset facial expressions to normal
            arm.joint_shapes[8] = (original_eyebrows[0], 4, 0, False, (0.3, 0.45, 0))  # Left eyebrow
            arm.joint_shapes[9] = (original_eyebrows[1], 4, 0, False, (-0.3, 0.45, 0))  # Right eyebrow

            mouth_width = 0.6
            laughing_mouth = create_manim_half_circle(mouth_width, color=BLACK, side='down')
            arm.joint_shapes[11] = (manim_to_vedo(laughing_mouth), 4, 0, False, (0, -0.3, 0))

            # Straighten legs as the robot descends
            child_arm2.angles[0] = np.radians(180)  # Straighten the left leg
            child_arm3.angles[0] = np.radians(180)  # Straighten the right leg
            child_arm2.FK()  
            child_arm3.FK() 

            # Rotate arms back to original position
            child_arm0.angles[0] = np.radians(180)
            child_arm0.FK()  
            child_arm1.angles[0] = np.radians(0)
            child_arm1.FK() 

        # Recalculate forward kinematics for all parts of the body and legs
        arm.FK(apply_initial_offset=False)
        child_arm2.FK(apply_initial_offset=False)
        child_arm3.FK(apply_initial_offset=False)
        child_arm1.FK(apply_initial_offset=False)
        child_arm0.FK(apply_initial_offset=False)

    # Perform the jump `n` times
    for _ in range(n):
        play_sound('sound/jump1.wav')

        start_time = time.time()
        
        # Execute a single jump over the duration
        while time.time() - start_time < duration:
            current_time = time.time() - start_time
            update_jump_position(current_time)  
            plt.remove("Assembly")  
            plt.add(draw_all_arms())  
            plt.render()
            time.sleep(time_per_frame)  
        
        # Reset angles to the initial angles after each jump to ensure legs and arms finish in the correct position
        child_arm2.angles = initial_leg1_angles.copy()
        child_arm3.angles = initial_leg2_angles.copy()
        child_arm1.angles = initial_child1_angles.copy()
        child_arm0.angles = initial_child0_angles.copy()

    
    # Reset the positions and angles to the original state after the last jump
    arm.offset = initial_offset.copy()
    child_arm2.offset = initial_leg1_offset.copy()
    child_arm3.offset = initial_leg2_offset.copy()
    child_arm1.offset = initial_child1_offset.copy()
    child_arm0.offset = initial_child0_offset.copy()

    # Ensure angles are set back to initial values after all jumps
    child_arm2.angles = initial_leg1_angles.copy()
    child_arm3.angles = initial_leg2_angles.copy()
    child_arm1.angles = initial_child1_angles.copy()
    child_arm0.angles = initial_child0_angles.copy()

    reset_face(arm)

    # Recalculate and redraw one last time to ensure everything is reset
    arm.FK()
    child_arm2.FK()
    child_arm3.FK()
    child_arm1.FK()
    child_arm0.FK()
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()




def dance(arm, duration=5.0, cycles=3):

    # Play the laughing sound
    play_sound('sound/dance5.wav')

    start_time = time.time()
    
    # Store original positions and angles
    original_angles = arm.angles.copy()
    original_offset = arm.offset.copy()
    child0_original_angles = child_arm0.angles.copy()
    child1_original_angles = child_arm1.angles.copy()
    child2_original_angles = child_arm2.angles.copy()  # Left leg
    child3_original_angles = child_arm3.angles.copy()  # Right leg


    child_arm2.angles[1] = child2_original_angles[1] - np.radians(30) # Left leg squeezing inward/outward
    child_arm3.angles[1] = child3_original_angles[1] + np.radians(30) # Left leg squeezing inward/outward

    
    # Dance animation: raising and lowering hands + squeezing legs
    frames = 30
    time_per_frame = duration / frames

    def update_dance_position(t):

        mouth_width = 0.5
        laughing_mouth = create_manim_half_circle(mouth_width, color=BLACK, side='down')
        arm.joint_shapes[11] = (manim_to_vedo(laughing_mouth), 4, 0, False, (0, -0.3, 0))


        # Control the frequency of the hand and leg movement with 'cycles'
        raise_angle = np.radians(45 * np.sin(2 * np.pi * cycles * t / duration))  # Smooth hand motion
        squeeze_angle = np.radians(15 * np.sin(2 * np.pi * cycles * t / duration))  # Smooth leg motion (squeezing)

        #head
        arm.angles[3] = original_angles[3] +  raise_angle 

        # Move the arms (child_arm0 and child_arm1) up and down
        child_arm0.angles[0] = child0_original_angles[0] + raise_angle  # Lift left arm
        child_arm0.angles[1] = child0_original_angles[1] + raise_angle + np.radians(30)  # Rotate second joint 


        child_arm1.angles[0] = child1_original_angles[0] + raise_angle  # Lift right arm
        child_arm1.angles[1] = child1_original_angles[1] + raise_angle + np.radians(30)  # Rotate second joint


        # Move the legs (child_arm2 and child_arm3) inward and outward to simulate squeezing
        child_arm2.angles[0] = child2_original_angles[0] + squeeze_angle  # Left leg squeezing inward/outward

        child_arm3.angles[0] = child3_original_angles[0] - squeeze_angle  # Right leg squeezing inward/outward


        # Update forward kinematics for all parts
        arm.FK(apply_initial_offset=False)
        child_arm0.FK(apply_initial_offset=False)
        child_arm1.FK(apply_initial_offset=False)
        child_arm2.FK(apply_initial_offset=False)
        child_arm3.FK(apply_initial_offset=False)
        
        # Redraw all arms
        plt.remove("Assembly")
        plt.add(draw_all_arms())
        plt.render()

    # Perform the dance animation for the specified duration
    while time.time() - start_time < duration:
        t = time.time() - start_time
        update_dance_position(t)
        time.sleep(time_per_frame)  # Control the frame rate

    # Reset to original angles and offsets after dancing
    arm.angles = original_angles
    arm.offset = original_offset
    child_arm0.angles = child0_original_angles
    child_arm1.angles = child1_original_angles
    child_arm2.angles = child2_original_angles
    child_arm3.angles = child3_original_angles


    # Reset face
    reset_face(arm)

    
    arm.FK()
    child_arm0.FK()
    child_arm1.FK()
    child_arm2.FK()
    child_arm3.FK()
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()



def stick_tongue_out(arm):
    RED = rgb_to_color([255, 0, 0])  

    # Step 1: Create a smaller, neutral mouth shape
    mouth_width = 5
    neutral_mouth = create_manim_arc(0.4, angle=PI/6, color=BLACK, stroke_width=mouth_width)
    arm.joint_shapes[11] = (manim_to_vedo(neutral_mouth), 4, 0, False, (0, -0.3, 0))  # Position the mouth

    # Step 2: Create the tongue shape as a small rounded rectangle inside the mouth
    tongue_shape = create_manim_half_circle(width=0.18, color=RED, side='down')
    tongue_initial_position = (0, -0.32, 0)  # Tongue starts slightly inside the mouth at the bottom center

    # Use a new index for the tongue to avoid conflicts
    new_tongue_index = len(arm.joint_shapes)  # This ensures we don't overwrite existing shapes
    arm.joint_shapes.append((manim_to_vedo(tongue_shape), 4, 0, False, tongue_initial_position))

    # Step 3: Blink the right eye by replacing it with a line
    right_eye_index = 4  # Index of the right eye
    closed_eye = create_manim_rectangle(0.24, 0.02, color=BLACK)  # Create a thin rectangle to represent a closed eye
    arm.joint_shapes[right_eye_index] = (manim_to_vedo(closed_eye), 4, 0, False, (0.3, 0.25, 0))  # Position the line as the "closed" eye

    right_eye2_index = 6  # Index of the right eye inner part
    arm.joint_shapes[right_eye2_index] = (manim_to_vedo(closed_eye), 4, 0, False, (0.3, 0.25, 0))  # Position the line as the "closed" eye

    # Step 4: Move the right eyebrow upward
    right_eyebrow_index = 9
    eyebrow_width = 0.4
    eyebrow_up_shift = 0.07  # Shift the eyebrow up

    # Recreate the right eyebrow higher on the face
    right_eyebrow = create_manim_downward_arc(eyebrow_width, color=BLACK)
    arm.joint_shapes[right_eyebrow_index] = (manim_to_vedo(right_eyebrow), 4, 0, False, (-0.3, 0.45 + eyebrow_up_shift, 0))

    # Render the initial positions of the face, mouth, and tongue
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()

    # Step 5: Animate all the movements together
    for i in range(5):
        # --- Tongue Animation ---
        tongue_scale_factor = 1 + i * 0.09  # Gradually increase the height to simulate the tongue sticking out
        arm.joint_shapes[new_tongue_index][0].scale([1, tongue_scale_factor, 1])  # Scale the tongue only vertically (downward)
        arm.joint_shapes[new_tongue_index][0].pos([0, -0.32 - i * 0.02, 0])  # Reposition the tongue slightly lower

        # --- Right Eyebrow Movement ---
        arm.joint_shapes[right_eyebrow_index][0].pos([-0.3, 0.45 + eyebrow_up_shift + i * 0.01, 0])  # Move the eyebrow slightly up

        # --- Eye Blink ---
        closed_eye = create_manim_rectangle(0.24, 0.02, color=BLACK)  # Update eye shape to be thinner
        arm.joint_shapes[right_eye_index] = (manim_to_vedo(closed_eye), 4, 0, False, (0.3, 0.25, 0))
        arm.joint_shapes[right_eye2_index] = (manim_to_vedo(closed_eye), 4, 0, False, (0.3, 0.25, 0))

        # Update and render the new positions and scales
        plt.remove("Assembly")
        plt.add(draw_all_arms())
        plt.render()
        time.sleep(0.1)

    # Step 6: Hold the expression for a short while before resetting
    time.sleep(2)

    # Step 7: Remove the tongue (make it disappear)
    del arm.joint_shapes[new_tongue_index]  # Remove the tongue from the joint_shapes list

    # Step 8: Reset the face back to its neutral state
    reset_face(arm)

    # Redraw the face in its neutral state
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()




def create_heart_shape(scale=0.01, color=(1, 0, 0)):
    t = np.linspace(0, 2 * np.pi, 100)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    points = np.column_stack((x, y, np.zeros_like(x))) * scale
    heart_shape = vd.Line(points).triangulate().extrude(0.02)
    heart_shape.color(color)
    return heart_shape




def love(arm, duration=5.0):
    start_time = time.time()
    scale_min, scale_max = 0.01, 0.015
    mouth_min, mouth_max = 0.55, 0.55
    animation_speed = 2.0

    left_eye_position = arm.joint_shapes[4][0].pos()
    right_eye_position = arm.joint_shapes[5][0].pos()

    # Create floating hearts once, outside the loop
    floating_hearts = [create_heart_shape(scale=0.005, color=(1, 0.5, 0.5)) for _ in range(4)]
    
    # Define custom initial positions for the hearts
    initial_positions = [
        [-0.8, 3.0, 0],  # Left side, higher
        [0, 4.0, 0],     # Center, even higher
        [0.8, 3.5, 0],    # Right side, higher
        [0.6, 2.8, 0]    # Right side, higher
    ]
    
    for heart, pos in zip(floating_hearts, initial_positions):
        heart.pos(pos)

    original_face_color = arm.joint_shapes[3][0].color()

    while time.time() - start_time < duration:
        t = (time.time() - start_time) * animation_speed
        scale_factor = scale_min + (scale_max - scale_min) * (0.5 * (1 + np.sin(t)))

        left_heart = create_heart_shape(scale=scale_factor)
        right_heart = create_heart_shape(scale=scale_factor)
        left_heart.pos(left_eye_position)
        right_heart.pos(right_eye_position)
        arm.joint_shapes[4] = (left_heart, 4, 0, False, (0.3, 0.25, 0))
        arm.joint_shapes[5] = (right_heart, 4, 0, False, (-0.3, 0.25, 0))

        mouth_width = mouth_min + (mouth_max - mouth_min) * (0.5 * (1 + np.sin(t)))
        new_mouth = create_manim_arc(mouth_width, angle=PI/6, color=BLACK, stroke_width=4)
        vedo_mouth = manim_to_vedo(new_mouth)
        arm.joint_shapes[11] = (vedo_mouth, 4, 0, False, (0, -0.3, 0))


        # Update positions of existing floating hearts
        for heart in floating_hearts:
            current_pos = heart.pos()
            new_y = current_pos[1] + 0.02  # Faster upward movement
            if new_y > 10:  # Reset when hearts go too high
                new_y = 7.0
            heart.pos([current_pos[0], new_y, current_pos[2]])

        plt.remove("Assembly")
        plt.add(draw_all_arms())
        for heart in floating_hearts:
            plt.add(heart)
        plt.render()

        time.sleep(0.05)


    # Clean up: remove floating hearts and reset the face
    for heart in floating_hearts:
        plt.remove(heart)
    reset_face(arm)
    arm.joint_shapes[3][0].color(original_face_color)

    # Re-render the scene without the floating hearts
    plt.remove("Assembly")
    plt.add(draw_all_arms())
    plt.render()

#============================================================================


def move_to_point(arm, target_point,currentMode, iterations=1000):

    IK_target = target_point

    for iteration in range(iterations):

        if currentMode == "Gradient-Descent":
            arm.inverse_kinematics_Gradient_Descent(IK_target)
        elif currentMode == "Gauss-Newton":
            arm.inverse_kinematics_Gauss_Newton(IK_target)
        else:
            print("Invalid currentMode. No action taken.")
            return  
            
        # Update the entire robot
        plt.remove("Assembly")
        plt.add(draw_all_arms())
        plt.render()

        distance = np.linalg.norm(arm.Jw[-1] - IK_target)
        if distance < 1e-2:
            break


currentARM = child_arm0
currentARMText = "RIGHT ARM"
currentMode = "Gauss-Newton"

  
D0 = child_arm0.angles.copy()
D1 = child_arm1.angles.copy()
D2 = child_arm2.angles.copy()  
D3 = child_arm3.angles.copy()


def LeftButtonPress(evt, iterations=1000):
    global currentARM,  currentMode 

    IK_target = evt.picked3d
    
    if IK_target is None:
        print("No 3D point was picked. Please try again.")
        return
    
    print(f'IK_target is {IK_target}')
    
    plt.remove("Sphere")
    plt.add(vd.Sphere(pos=IK_target, r=0.05, c='b'))
    
    for iteration in range(iterations):
        if currentMode == "Gradient-Descent":
            currentARM.inverse_kinematics_Gradient_Descent(IK_target)
        elif currentMode == "Gauss-Newton":
            currentARM.inverse_kinematics_Gauss_Newton(IK_target)
        else:
            print("Invalid currentMode. No action taken.")
            return
        
        plt.remove("Assembly")
        plt.add(draw_all_arms())
        plt.render()
        
        distance = np.linalg.norm(currentARM.Jw[-1] - IK_target)
        if distance < 1e-2:
            break



def onKeyPress(evt):
    global currentARM, currentARMText, currentMode, D0,D1,D2,D3

    reset_face(arms[0])

    child_arm0.angles = D0.copy()
    child_arm1.angles = D1.copy()
    child_arm2.angles = D2.copy()
    child_arm3.angles = D3.copy()

    if evt.keypress in ['i','I']:
        wave(child_arm1,arms[0]) 
    elif evt.keypress in ['b','B']:
        blink(arms[0], [4, 5, 6, 7])  
    elif evt.keypress in ['h','H']:
        laughing(arms[0])
    elif evt.keypress in ['c','C']:
        cry(arms[0])  
    elif evt.keypress in ['a','A']:
        angry(arms[0])  
    elif evt.keypress in ['s','S']:  
        surprised(arms[0])
    elif evt.keypress in ['n','N']:  
        yawning(arms[0])
    elif evt.keypress in ['j','J']:  
        jump(arms[0], height=1.5, duration=1.0)
    elif evt.keypress in ['d','D']: 
        dance(arms[0], duration=5.0, cycles=2) 
    elif evt.keypress in ['g','G']:
        stick_tongue_out(arms[0])   
    elif evt.keypress in ['v','V']:  
        love(arms[0], duration=5.0)

    #----------------
    elif evt.keypress in ['0']:
        currentARM = child_arm0
        currentARMText = "RIGHT ARM"
        currentARM_Text.text("Current Arm: " + currentARMText)
    elif evt.keypress in ['1']:
        currentARM = child_arm1
        currentARMText = "LEFT ARM"
        currentARM_Text.text("Current Arm: " + currentARMText)
    elif evt.keypress in ['2']:
        currentARM = child_arm2
        currentARMText = "LEFT LEG"
        currentARM_Text.text("Current Arm: " + currentARMText)
    elif evt.keypress in ['3']:
        currentARM = child_arm3
        currentARMText = "RIGHT LEG"
        currentARM_Text.text("Current Arm: " + currentARMText)

    #-----------------
    elif evt.keypress in ['4']:
        currentMode = "Gradient-Descent"
        currentcurrentMode_Text.text("Current Mode: " + currentMode)
    elif evt.keypress in ['5']:
        currentMode = "Gauss-Newton"
        currentcurrentMode_Text.text("Current Mode: " + currentMode)

    #-----------------
    elif evt.keypress in ['6']:
        move_to_point(child_arm1, [2.6, 1.3, 0],"Gradient-Descent") 
    elif evt.keypress in ['7']:
        move_to_point(child_arm1,[-1, 2, 0],"Gauss-Newton")




# %% Initialize plotter and add elements
plt = vd.Plotter()
plt += draw_all_arms()

plt += vd.Sphere(pos = IK_target, r=0.05, c='b').draggable(True)
plane = vd.Plane(s=[10,10]) 
plt.add(plane)

textSize = 0.6
expl_Text0 = vd.Text2D("This robot can: I - Wave, D - Dance, B - Blink, H - Laugh, C - Cry, A - Show anger, S - Show surprise, N - Yawn, J - Jump, G - Stick its tongue out, V - Show love.", pos=(0.05,0.99), s=textSize)
plt.add(expl_Text0)

expl_Text1 = vd.Text2D("Using the mouse, you can select a point, and one arm will move. You can choose the arm by pressing:", pos=(0.05,0.95), s=textSize)
plt.add(expl_Text1)

intro_Text0 = vd.Text2D("0 -> RIGHT ARM , 1 -> LEFT ARM", pos=(0.05,0.90), s=textSize)
intro_Text1 = vd.Text2D("2 -> LEFT LEG , 3 -> RIGHT LEG", pos=(0.05,0.85), s=textSize)
plt.add(intro_Text0)
plt.add(intro_Text1)
currentARM_Text = vd.Text2D("CURRENT ARM : " + currentARMText, pos=(0.05,0.80), s=textSize)
plt.add(currentARM_Text)


intro_Text2 = vd.Text2D("Press: 4 -> Gradient-Descent Mode", pos=(0.05,0.65), s=textSize)
intro_Text3 = vd.Text2D("       5 -> Gauss-Newton Mode", pos=(0.05,0.60), s=textSize)
plt.add(intro_Text2)
plt.add(intro_Text3)
currentcurrentMode_Text = vd.Text2D("CURRENT MODE : " + currentMode, pos=(0.05,0.55), s=textSize)
plt.add(currentcurrentMode_Text)

plt.remove("Assembly")
plt.add(draw_all_arms())
plt.render()

plt.add_callback('LeftButtonPress', LeftButtonPress)

plt.add_callback('KeyPress', onKeyPress)
plt.user_mode('2d').show(zoom="tightest")

plt.close()