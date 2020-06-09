#!/usr/bin/env python
import math
import numpy as np
import string
import argparse

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.iteritems())
_NEXT_AXIS = [1, 2, 0, 1]
_EPS = np.finfo(np.float64).eps * 4.0

def euler_from_matrix(matrix, axes='sxzx'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> np.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (np.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not np.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

# Node
class Node:
    def __init__(self, root=False):
        self.name = None
        self.channels = []
        self.offset = (0,0,0)
        self.children = []
        self.is_root = root

    def isEndSite(self):
        return len(self.children)==0

    def to_json(self):
        return {
            "name": self.name,
            "channels": self.channels,
            "offset": self.offset,
            "is_root": self.is_root,
            "children": [child.to_json() for child in self.children]
        }
    
    def __repr__(self):
        return str(self.to_json())
    
# BVHReader
class BVHReader:
    def __init__(self, filename):
        self.filename = filename # BVH filename
        self.tokenlist = [] # A list of unprocessed tokens (strings)       
        self.linenr = 0  # The current line number

        # Root node
        self.root = None
        self.nodestack = []

        # Total number of channels
        self.numchannels = 0
        
        # Motion
        self.all_motions = []
        self.dt = 1
        self.num_motions = 1

        self.counter = 0
        self.this_motion = None

        self.scaling_factor = 0.1
        
        # Read file
        self.fhandle = open(self.filename, 'r')
        self.readHierarchy()
        self.readMotion()
        
    # Tokenization function
    def readLine(self):
        """Return the next line."""
        s = self.fhandle.readline()
        self.linenr += 1
        if s=="":
            raise None # End of file
        return s
        
    def token(self):
        """Return the next token."""
        # Are there still some tokens left? then just return the next one
        if self.tokenlist!=[]:
            tok = self.tokenlist[0]
            self.tokenlist = self.tokenlist[1:]
            return tok

        # Read a new line
        s = self.readLine()
        self.tokenlist = s.strip().split()
        return self.token()
    
    def intToken(self):
        """Return the next token which must be an int.
        """
        tok = self.token()
        try:
            return int(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Integer expected, got '%s' instead"%(self.linenr, tok))

    def floatToken(self):
        """Return the next token which must be a float.
        """
        tok = self.token()
        try:
            return float(tok)
        except ValueError:
            raise SyntaxError("Syntax error in line %d: Float expected, got '%s' instead"%(self.linenr, tok))

    ###
    # Read Hierarchy
    ###
    def readHierarchy(self):
        """Read the skeleton hierarchy to self.nodestack and store root in self.root
        """
        tok = self.token()
        if tok!="HIERARCHY":
            raise SyntaxError("Syntax error in line %d: 'HIERARCHY' expected, got '%s' instead"%(self.linenr, tok))

        tok = self.token()
        if tok!="ROOT":
            raise SyntaxError("Syntax error in line %d: 'ROOT' expected, got '%s' instead"%(self.linenr, tok))

        self.root = Node(root=True)
        self.nodestack.append(self.root)
        self.readNode()
        
        # Handler OnHierarchy
        self.scaling_factor = 0.1/self.root.children[0].children[0].offset[0]
        
    # readNode
    def readNode(self):
        """Recursive function to recursively read the data for a node.
        """
        # Read the node name (or the word 'Site' if it was a 'End Site' node)
        name = self.token()
        self.nodestack[-1].name = name
        
        tok = self.token()
        if tok!="{":
            raise SyntaxError("Syntax error in line %d: '{' expected, got '%s' instead"%(self.linenr, tok))

        while True:
            tok = self.token()
            if tok=="OFFSET":
                x, y, z = self.floatToken(), self.floatToken(), self.floatToken()
                self.nodestack[-1].offset = (x,y,z)
            elif tok=="CHANNELS":
                n = self.intToken()
                channels = []
                for i in range(n):
                    tok = self.token()
                    if tok not in ["Xposition", "Yposition", "Zposition",
                                  "Xrotation", "Yrotation", "Zrotation"]:
                        raise SyntaxError("Syntax error in line %d: Invalid channel name: '%s'"%(self.linenr, tok))                        
                    channels.append(tok)
                self.numchannels += len(channels)
                self.nodestack[-1].channels = channels
            elif tok=="JOINT":
                node = Node()
                self.nodestack[-1].children.append(node)
                self.nodestack.append(node)
                self.readNode()
            elif tok=="End":
                node = Node()
                self.nodestack[-1].children.append(node)
                self.nodestack.append(node)
                self.readNode()
            elif tok=="}":
                if self.nodestack[-1].isEndSite():
                    self.nodestack[-1].name = "End Site"
                self.nodestack.pop()
                break
            else:
                raise SyntaxError("Syntax error in line %d: Unknown keyword '%s'"%(self.linenr, tok))        
        
    ###
    # Read Motion
    ###        
    def readMotion(self):
        """Read the motion samples and stores to self.all_motions
        """
        # No more tokens (i.e. end of file)? Then just return 
        tok = self.token()
        if not tok:
            return
        
        if tok!="MOTION":
            raise SyntaxError("Syntax error in line %d: 'MOTION' expected, got '%s' instead"%(self.linenr, tok))

        # Read the number of frames
        tok = self.token()
        if tok!="Frames:":
            raise SyntaxError("Syntax error in line %d: 'Frames:' expected, got '%s' instead"%(self.linenr, tok))

        frames = self.intToken()

        # Read the frame time
        tok = self.token()
        if tok!="Frame":
            raise SyntaxError("Syntax error in line %d: 'Frame Time:' expected, got '%s' instead"%(self.linenr, tok))
        tok = self.token()
        if tok!="Time:":
            raise SyntaxError("Syntax error in line %d: 'Frame Time:' expected, got 'Frame %s' instead"%(self.linenr, tok))

        dt = self.floatToken()
        
        # Handler OnMotion
        self.dt = dt
        self.num_motions = frames
        
        # Read the channel values
        for i in range(frames):
            s = self.readLine()
            a = s.split()
            if len(a)!=self.numchannels:
                raise SyntaxError("Syntax error in line %d: %d float values expected, got %d instead"%(self.linenr, self.numchannels, len(a)))
            values = map(lambda x: float(x), a)
            
            # Handler OnFrame
            self.all_motions.append(list(values))

    # BroadcastRootJoint
    def broadcastRootJoint(self, root):
        if root.isEndSite():
            return

        num_channels = len(root.channels)
        flag_trans = 0
        flag_rot = 0
        rx, ry, rz = 0, 0, 0
        rot_mat = np.array([[1.,0.,0.,0.], 
                            [0.,1.,0.,0.], 
                            [0.,0.,1.,0.], 
                            [0.,0.,0.,1.]])

        for channel in root.channels:
            keyval = self.this_motion[self.counter]
            if channel == "Xposition":
                flag_trans = True
                x = keyval
            elif(channel == "Yposition"):
                flag_trans = True
                y = keyval
            elif(channel == "Zposition"):
                flag_trans = True
                z = keyval
            elif(channel == "Xrotation"):
                flag_rot = True
                xrot = keyval
                theta = math.radians(xrot)
                rx = theta
                c = math.cos(theta)
                s = math.sin(theta)
                rot_mat_x = np.array([[1.,0.,0.,0.], 
                                        [0., c,-s,0.], 
                                        [0., s, c,0.], 
                                        [0.,0.,0.,1.]])
                rot_mat = np.matmul(rot_mat, rot_mat_x)
            elif(channel == "Yrotation"):
                flag_rot = True
                yrot = keyval
                theta = math.radians(yrot)
                ry = theta
                c = math.cos(theta)
                s = math.sin(theta)
                rot_mat_y = np.array([[ c,0., s,0.],
                                        [0.,1.,0.,0.],
                                        [-s,0., c,0.],
                                        [0.,0.,0.,1.]])
                rot_mat = np.matmul(rot_mat, rot_mat_y)

            elif(channel == "Zrotation"):
                flaisRootg_rot = True
                zrot = keyval
                theta = math.radians(zrot)
                ry = theta
                c = math.cos(theta)
                s = math.sin(theta)
                rot_mat_z = np.array([[ c,-s,0.,0.],
                                        [ s, c,0.,0.],
                                        [0.,0.,1.,0.],
                                        [0.,0.,0.,1.]])
                rot_mat = np.matmul(rot_mat, rot_mat_z)
            else:
                return
            self.counter += 1
        
        if flag_trans:
            temp_trans = (self.scaling_factor * (x + root.offset[0]), 
                          self.scaling_factor * (y + root.offset[1]), 
                          self.scaling_factor * (z + root.offset[2]))
        else:
            temp_trans = (self.scaling_factor * (root.offset[0]), 
                          self.scaling_factor * (root.offset[1]), 
                          self.scaling_factor * (root.offset[2]))

        print('tx, ty, tz', temp_trans)
        print('rx, ry, rz', rx, ry, rz)

        # Transform rotation to Pepper coordinate system
        rx, ry, rz = euler_from_matrix(rot_mat, axes='sxyz')

        print('r\'x, r\'y, r\'z', rx, ry, rz)
        print()

        for each_child in root.children:
            self.broadcastRootJoint(each_child)

    def broadcast(self, loop=False):
        for ind in range(self.num_motions):
            self.counter = 0
            self.this_motion = self.all_motions[ind]
            self.broadcastRootJoint(self.root)

if __name__ == "__main__":
    bvh_file = '../cmuconvert-mb2/01/01_01.bvh'
    bvh_test = BVHReader(bvh_file)
    
    print(bvh_test.root)
    print(len(bvh_test.all_motions))
    
    bvh_test.broadcast()
    
