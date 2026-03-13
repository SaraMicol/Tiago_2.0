#!/usr/bin/env python3
"""
Object Manager Service - Semantic and Spatial Tracking of Perceived Objects
Tracks objects and automatically transitions from EXPLORATION to TRACKING when
an object is seen again in a different position.
"""
import rclpy, json, os, time, logging, threading
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy
import numpy as np
from openai import OpenAI
from lost3dsg.msg import ObjectDescriptionArray, Bbox3dArray
from lost3dsg.srv import ObjectTrackingService
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool
from object_info import Object
from debug_utils import TrackingLogger
from world_model import wm
import gensim.downloader as api
from utils import *
from nlp_utils import *
from datetime import datetime
from cv_utils import *
from map_database import MapDatabase

# =============  EXPLORATION PARAMETERS =============
EXPLORATION_IOU_THRESHOLD = 0.18
SIM_THRESHOLD = 0.75
TRACKING_IOU_THRESHOLD = 0.3
VOLUME_EXPANSION_RATIO = 0.01
EXPLORATION_FRAME_LIMIT = 10  # Numero di frame in exploration prima di passare a tracking

# Load OpenAI API key
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
PROJECT_ROOT = current_dir.split('/install/')[0] if '/install/' in current_dir else os.path.abspath(os.path.join(current_dir, "../.."))

with open(os.path.join(PROJECT_ROOT, "perception_module", "api.txt"), "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)
world2vec = api.load('word2vec-google-news-300')

# Setup file logger and project paths
log_dir = os.path.join(PROJECT_ROOT, "output")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"object_manager_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))

# Setup module logger
module_logger = logging.getLogger('object_manager_module')
module_logger.setLevel(logging.DEBUG)
module_logger.addHandler(file_handler)

# Dedicated tracking file for important operations
tracking_log_file = os.path.join(log_dir, f"tracking_operations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
tracking_logger = TrackingLogger(tracking_log_file)

# ============= HELPER FUNCTIONS =============

def create_object_key(label, material, color, description):
    """Create a unique key for an object based on attributes."""
    key_dict = {
        "label": label if label else "",
        "material": material if material else "",
        "color": color if color else "",
        "description": description if description else ""
    }
    return json.dumps(key_dict, sort_keys=True)


def compute_pov_volume(bboxes_list, expansion_ratio=VOLUME_EXPANSION_RATIO):
    """Compute the POV volume that contains all detections."""
    if not bboxes_list:
        return None

    MAX_VOLUME_THRESHOLD = 0.5
    BBOX_REDUCTION_RATIO = 0.30

    def shrink_bbox(bbox, ratio):
        x_center = (bbox["x_min"] + bbox["x_max"]) / 2.0
        y_center = (bbox["y_min"] + bbox["y_max"]) / 2.0
        z_center = (bbox["z_min"] + bbox["z_max"]) / 2.0

        x_size = (bbox["x_max"] - bbox["x_min"]) * (1.0 - ratio)
        y_size = (bbox["y_max"] - bbox["y_min"]) * (1.0 - ratio)
        z_size = (bbox["z_max"] - bbox["z_min"]) * (1.0 - ratio)

        return {
            "x_min": x_center - x_size / 2.0,
            "x_max": x_center + x_size / 2.0,
            "y_min": y_center - y_size / 2.0,
            "y_max": y_center + y_size / 2.0,
            "z_min": z_center - z_size / 2.0,
            "z_max": z_center + z_size / 2.0
        }

    def bbox_volume(bbox):
        return ((bbox["x_max"] - bbox["x_min"]) *
                (bbox["y_max"] - bbox["y_min"]) *
                (bbox["z_max"] - bbox["z_min"]))

    processed_bboxes = []
    for bbox in bboxes_list:
        vol = bbox_volume(bbox)
        if vol > MAX_VOLUME_THRESHOLD:
            processed_bbox = shrink_bbox(bbox, BBOX_REDUCTION_RATIO)
        else:
            processed_bbox = bbox
        processed_bboxes.append(processed_bbox)

    x_min = min(bbox["x_min"] for bbox in processed_bboxes)
    x_max = max(bbox["x_max"] for bbox in processed_bboxes)
    y_min = min(bbox["y_min"] for bbox in processed_bboxes)
    y_max = max(bbox["y_max"] for bbox in processed_bboxes)
    z_min = min(bbox["z_min"] for bbox in processed_bboxes)
    z_max = max(bbox["z_max"] for bbox in processed_bboxes)

    x_size = x_max - x_min
    y_size = y_max - y_min
    z_size = z_max - z_min

    x_expansion = x_size * expansion_ratio
    y_expansion = y_size * expansion_ratio
    z_expansion = z_size * expansion_ratio

    MIN_EXPANSION = 0.1
    x_expansion = max(x_expansion, MIN_EXPANSION)
    y_expansion = max(y_expansion, MIN_EXPANSION)
    z_expansion = max(z_expansion, MIN_EXPANSION)

    pov_z_min = z_min - z_expansion
    pov_z_max = z_max

    return {
        "x_min": x_min - x_expansion,
        "x_max": x_max + x_expansion,
        "y_min": y_min - y_expansion,
        "y_max": y_max + y_expansion,
        "z_min": pov_z_min,
        "z_max": pov_z_max
    }


def expand_bbox_for_search(bbox, expansion_ratio=VOLUME_EXPANSION_RATIO):
    """Expand a bounding box proportionally to its size."""
    x_size = bbox["x_max"] - bbox["x_min"]
    y_size = bbox["y_max"] - bbox["y_min"]
    z_size = bbox["z_max"] - bbox["z_min"]
    
    x_expansion = x_size * expansion_ratio
    y_expansion = y_size * expansion_ratio
    z_expansion = z_size * expansion_ratio
    
    return {
        "x_min": bbox["x_min"] - x_expansion,
        "x_max": bbox["x_max"] + x_expansion,
        "y_min": bbox["y_min"] - y_expansion,
        "y_max": bbox["y_max"] + y_expansion,
        "z_min": bbox["z_min"] - z_expansion,
        "z_max": bbox["z_max"] + z_expansion
    }


def bbox_centroid_in_volume(bbox, volume):
    """Check whether the centroid of a bounding box lies inside a volume."""
    if bbox is None:
        return False

    centroid_x = (bbox["x_min"] + bbox["x_max"]) / 2.0
    centroid_y = (bbox["y_min"] + bbox["y_max"]) / 2.0
    centroid_z = (bbox["z_min"] + bbox["z_max"]) / 2.0

    is_inside = (
        volume["x_min"] <= centroid_x <= volume["x_max"] and
        volume["y_min"] <= centroid_y <= volume["y_max"] and
        volume["z_min"] <= centroid_z <= volume["z_max"]
    )

    return is_inside


def save_persistent_perceptions(node):
    """Save persistent_perceptions to JSON."""
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "persistent_perception.json")

    data = []
    for obj in wm.persistent_perceptions:
        data.append({
            "label": obj.label,
            "description": obj.description,
            "color": obj.color,
            "material": obj.material,
            "shape": obj.shape,
            "bbox": obj.bbox
        })

    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)

    msg = f"Saved {len(data)} objects to persistent_perception.json"
    node.log_both('info', msg)
    labels = [obj["label"] for obj in data]
    node.log_both('info', f"Saved object labels: {labels}")


def save_scene_graph(node, step, is_exploration=False):
    """Generate and save a 3D scene graph image for the current step."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    objects = []
    for obj in wm.persistent_perceptions:
        objects.append({
            "label": obj.label,
            "color": obj.color,
            "material": obj.material,
            "bbox": obj.bbox
        })
    
    prefix = "exploration" if is_exploration else "tracking"
    json_path = os.path.join(output_dir, f"scene_graph_{prefix}_{step:03d}.json")
    with open(json_path, 'w') as f:
        json.dump({
            "step": step,
            "mode": prefix,
            "num_objects": len(objects),
            "objects": objects
        }, f, indent=2)
    node.log_both('info', f"[SCENE GRAPH] Saved JSON: {json_path}")

    if len(objects) == 0:
        node.log_both('info', f"[SCENE GRAPH] No objects to visualize for step {step}")
        return

    fig = plt.figure(figsize=(14, 10), facecolor='#F9F7F7')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#F9F7F7')

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    centroids = []
    for obj in objects:
        bbox = obj['bbox']
        centroid = np.array([
            (bbox['x_min'] + bbox['x_max']) / 2,
            (bbox['y_min'] + bbox['y_max']) / 2,
            (bbox['z_min'] + bbox['z_max']) / 2
        ])
        centroids.append(centroid)
    
    centroids = np.array(centroids)
    scene_center = centroids.mean(axis=0)
    z_max = max(c[2] for c in centroids) + 0.3
    root_pos = np.array([scene_center[0], scene_center[1], z_max])

    ax.scatter(*root_pos, s=500, c='#4A6572', marker='o', zorder=10)
    ax.text(root_pos[0], root_pos[1], root_pos[2] + 0.08, 'SCENE',
            ha='center', va='bottom', fontsize=13, fontweight='bold', color='#344955')

    for i, obj in enumerate(objects):
        centroid = centroids[i]
        ax.plot3D([root_pos[0], centroid[0]],
                  [root_pos[1], centroid[1]],
                  [root_pos[2], centroid[2]],
                  color='#9DB2BF', linewidth=1.5, alpha=0.5)

        ax.scatter(*centroid, s=400, c='#7EB5D6', marker='o', zorder=10)
        ax.text(centroid[0], centroid[1], centroid[2] + 0.1,
                obj['label'],
                ha='center', va='bottom', fontsize=10, fontweight='semibold',
                color='#2C3E50')

    ax.set_xlabel('X (m)', fontsize=11, color='#526D82')
    ax.set_ylabel('Y (m)', fontsize=11, color='#526D82')
    ax.set_zlabel('Z (m)', fontsize=11, color='#526D82')

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"scene_graph_{prefix}_{step:03d}.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    node.log_both('info', f"[SCENE GRAPH] Saved: {output_path}")


def save_uncertain_objects(node):
    """Save uncertain_objects to a text file."""
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "uncertain_objects.txt")

    with open(save_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"UNCERTAIN OBJECTS - Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        if not node.uncertain_objects:
            f.write("No uncertain objects at the moment.\n")
        else:
            f.write(f"Total uncertain objects: {len(node.uncertain_objects)}\n\n")

            for i, obj in enumerate(node.uncertain_objects, 1):
                f.write(f"{i}. {obj.label}\n")
                f.write(f"   Description: {obj.description}\n")
                f.write(f"   Color: {obj.color}\n")
                f.write(f"   Material: {obj.material}\n")

                if obj.bbox:
                    x_center = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
                    y_center = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
                    z_center = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0

                    x_size = obj.bbox["x_max"] - obj.bbox["x_min"]
                    y_size = obj.bbox["y_max"] - obj.bbox["y_min"]
                    z_size = obj.bbox["z_max"] - obj.bbox["z_min"]

                    f.write(f"   Center position: X={x_center:.3f}, Y={y_center:.3f}, Z={z_center:.3f}\n")
                    f.write(f"   Dimensions: {x_size:.3f}m x {y_size:.3f}m x {z_size:.3f}m\n")
                else:
                    f.write(f"   Bbox: NOT AVAILABLE\n")

                f.write("\n" + "-" * 80 + "\n\n")

    node.log_both('info', f"Saved {len(node.uncertain_objects)} uncertain objects")


def publish_persistent_bboxes(node, wm, pub):
    """Publish persistent bounding boxes as markers."""
    marker_array = MarkerArray()
    for i, obj in enumerate(wm.persistent_perceptions):
        if obj.bbox is None:
            continue
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
        marker.pose.position.y = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
        marker.pose.position.z = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
        marker.scale.x = obj.bbox['x_max'] - obj.bbox['x_min']
        marker.scale.y = obj.bbox['y_max'] - obj.bbox['y_min']
        marker.scale.z = obj.bbox['z_max'] - obj.bbox['z_min']
        marker.color.a = 0.5
        marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        marker_array.markers.append(marker)
    pub.publish(marker_array)


def publish_persistent_centroids(node, wm, pub):
    """Publish persistent object centroids as markers."""
    marker_array = MarkerArray()
    for i, obj in enumerate(wm.persistent_perceptions):
        if obj.bbox is None:
            continue
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
        marker.pose.position.y = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
        marker.pose.position.z = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        marker_array.markers.append(marker)
    pub.publish(marker_array)


def publish_uncertain_bboxes(node, uncertain_objects, pub):
    """Publish uncertain bounding boxes as markers."""
    marker_array = MarkerArray()
    for i, obj in enumerate(uncertain_objects):
        if obj.bbox is None:
            continue
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
        marker.pose.position.y = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
        marker.pose.position.z = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
        marker.scale.x = obj.bbox['x_max'] - obj.bbox['x_min']
        marker.scale.y = obj.bbox['y_max'] - obj.bbox['y_min']
        marker.scale.z = obj.bbox['z_max'] - obj.bbox['z_min']
        marker.color.a = 0.5
        marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        marker_array.markers.append(marker)
    pub.publish(marker_array)


def publish_uncertain_centroids(node, uncertain_objects, pub):
    """Publish uncertain object centroids as markers."""
    marker_array = MarkerArray()
    for i, obj in enumerate(uncertain_objects):
        if obj.bbox is None:
            continue
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = node.get_clock().now().to_msg()
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
        marker.pose.position.y = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
        marker.pose.position.z = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r, marker.color.g, marker.color.b = 1.0, 0.5, 0.0
        marker_array.markers.append(marker)
    pub.publish(marker_array)


def publish_pov_volume(node, pov_volume, pub):
    """Publish POV volume as a marker."""
    marker_array = MarkerArray()
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = (pov_volume['x_min'] + pov_volume['x_max']) / 2.0
    marker.pose.position.y = (pov_volume['y_min'] + pov_volume['y_max']) / 2.0
    marker.pose.position.z = (pov_volume['z_min'] + pov_volume['z_max']) / 2.0
    marker.scale.x = pov_volume['x_max'] - pov_volume['x_min']
    marker.scale.y = pov_volume['y_max'] - pov_volume['y_min']
    marker.scale.z = pov_volume['z_max'] - pov_volume['z_min']
    marker.color.a = 0.2
    marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
    marker_array.markers.append(marker)
    pub.publish(marker_array)


class ObjectManagerService(Node):
    def __init__(self):
        super().__init__('object_tracking_service_node')

        self.file_logger = module_logger
        self.file_logger.info("=== ObjectManagerService Initialized ===")
        self.get_logger().info(f"Log saved in: {log_file}")

        self.exploration_mode = True
        self.seen_again = False
        self.latest_bboxes = {}
        self.latest_fov_volume = None
        self.uncertain_objects = []
        self.tracking_step_counter = 0
        self.exploration_step_counter = 0
        self.exploration_frame_counter = 0
        self.db = MapDatabase(db_path=os.path.join(log_dir, "tiago_temporal_map_1.db"))
        self.robot_has_moved = False

        # NUOVO
        self.latest_descriptions = None
        self.latest_bboxes_msg = None

        qos_latch = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        qos_standard = QoSProfile(depth=10)
        
        # Publishers
        self.persistent_bbox_pub = self.create_publisher(MarkerArray, '/persistent_bbox', qos_latch)
        self.persistent_centroids_pub = self.create_publisher(MarkerArray, '/persistent_centroids', qos_latch)
        self.considered_volume_pub = self.create_publisher(MarkerArray, '/considered_volume', qos_standard)
        self.uncertain_bboxes_pub = self.create_publisher(MarkerArray, '/uncertain_object', qos_standard)
        self.uncertain_centroids_pub = self.create_publisher(MarkerArray, '/uncertain_centroids', qos_standard)
        self.tracking_activated_pub = self.create_publisher(Bool, '/tracking_mode_activated', qos_standard)

        # Service server (rimane per compatibilità)
        self.srv = self.create_service(
            ObjectTrackingService,
            'object_tracking_service',
            self.object_tracking_callback
        )
        self.get_logger().info('Object Tracking Service ready')

        # Subscribers
        self.create_subscription(Bool, "/robot_movement_detected", self.movement_callback, qos_standard)

        # NUOVI - sottoscrizione diretta ai topic di detection_node
        self.create_subscription(
            ObjectDescriptionArray,
            '/object_descriptions',
            self._descriptions_callback,
            qos_standard
        )
        self.create_subscription(
            Bbox3dArray,
            '/bbox_3d',
            self._bboxes_callback,
            qos_standard
        )

        self._bbox_timer = self.create_timer(2.0, self.periodic_bbox_publisher)
        
    def movement_callback(self, msg):
        """Callback to receive robot movement notification."""
        self.log_both('info', f"[MOVEMENT] Robot moved: {msg.data}")
        if msg.data and not self.robot_has_moved:
            self.robot_has_moved = True
            self.log_both('warn', "[MOVEMENT] ✓ Robot has moved - ready for tracking transition")

    def log_both(self, level, message):
        """Log to both ROS and file logger."""
        if level == 'info':
            self.get_logger().info(message)
            self.file_logger.info(message)
        elif level == 'warn':
            self.get_logger().warn(message)
            self.file_logger.warning(message)
        elif level == 'error':
            self.get_logger().error(message)
            self.file_logger.error(message)
        elif level == 'debug':
            self.get_logger().debug(message)
            self.file_logger.debug(message)

        if level in ['info', 'warn']:
            prefix = "[INFO] " if level == 'info' else "[WARN] "
            print(f"{prefix}{message}")

    def check_tracking_transition(self, label_base, color, material, description_embedding, bbox):
        """
        Check if an object was seen before in a different position.
        If yes, automatically transition to TRACKING mode.
        
        Returns:
            tuple: (transition_triggered, obj, distance_moved)
        """
        for obj in wm.persistent_perceptions:
            obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label
            
            if not hasattr(obj, "embedding"):
                obj.embedding = get_embedding(client, obj.description)
            if obj.embedding is None:
                continue
            
            if description_embedding is None:
                continue
            
            similarity = lost_similarity(world2vec, label_base, obj_label_base, color, obj.color,
                                       material, obj.material, description_embedding, obj.embedding)
            
            if similarity <= SIM_THRESHOLD:
                continue
            
            if obj.bbox is None:
                continue
            
            old_x = (obj.bbox['x_min'] + obj.bbox['x_max']) / 2.0
            old_y = (obj.bbox['y_min'] + obj.bbox['y_max']) / 2.0
            old_z = (obj.bbox['z_min'] + obj.bbox['z_max']) / 2.0
            new_x = (bbox['x_min'] + bbox['x_max']) / 2.0
            new_y = (bbox['y_min'] + bbox['y_max']) / 2.0
            new_z = (bbox['z_min'] + bbox['z_max']) / 2.0
            
            distance = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2 + (new_z - old_z)**2)
            iou = compute_iou_3d(bbox, obj.bbox)
            
            # Different position = trigger tracking
            if distance > 0.2 and iou < EXPLORATION_IOU_THRESHOLD:
                self.log_both('warn', f"🔴 [TRACKING TRANSITION] Object '{obj.label}' seen at different position!")
                self.log_both('warn', f"   Old position: ({old_x:.2f}, {old_y:.2f}, {old_z:.2f})")
                self.log_both('warn', f"   New position: ({new_x:.2f}, {new_y:.2f}, {new_z:.2f})")
                self.log_both('warn', f"   Distance moved: {distance:.2f}m | IoU: {iou:.3f}")
                return True, obj, distance
        
        return False, None, 0.0

    def object_tracking_callback(self, request, response):
        """
        Service callback: process descriptions and bboxes.
        Automatically transitions to TRACKING when object seen in new position.
        """
        if not self.exploration_mode:
            self.tracking_step_counter += 1
        
        in_exploration = self.exploration_mode
        mode_str = "EXPLORATION" if in_exploration else f"TRACKING STEP {self.tracking_step_counter}"

        current_perception_objects = []
        objects_modified = False
        tracking_activated = False

        if in_exploration:
            self.exploration_frame_counter += 1

        # Process bboxes first
        self.latest_bboxes = {}
        
        if request.bboxes.fov_x_max != 0 or request.bboxes.fov_y_max != 0 or request.bboxes.fov_z_max != 0:
            self.latest_fov_volume = {
                "x_min": request.bboxes.fov_x_min,
                "x_max": request.bboxes.fov_x_max,
                "y_min": request.bboxes.fov_y_min,
                "y_max": request.bboxes.fov_y_max,
                "z_min": request.bboxes.fov_z_min,
                "z_max": request.bboxes.fov_z_max
            }

        for box in request.bboxes.boxes:
            x_min, x_max = min(box.x_min, box.x_max), max(box.x_min, box.x_max)
            y_min, y_max = min(box.y_min, box.y_max), max(box.y_min, box.y_max)
            z_min, z_max = min(box.z_min, box.z_max), max(box.z_min, box.z_max)

            bbox_data = {
                "x_min": x_min, "x_max": x_max,
                "y_min": y_min, "y_max": y_max,
                "z_min": z_min, "z_max": z_max
            }
            temp_key = create_object_key(box.label, "", "", "")
            self.latest_bboxes[temp_key] = {
                "bbox": bbox_data,
                "label": box.label,
                "color": "", "material": "", "description": ""
            }

        # Process descriptions
        for description in request.descriptions.descriptions:
            label = description.label
            label_base = label.split('#')[0] if '#' in label else label
            color = description.color
            material = description.material
            description_text = description.description

            description_embedding = get_embedding(client, description_text)

            old_key = create_object_key(label, "", "", "")
            if old_key not in self.latest_bboxes:
                continue

            bbox = self.latest_bboxes[old_key]["bbox"]
            new_key = create_object_key(label, material, color, description_text)
            
            del self.latest_bboxes[old_key]
            self.latest_bboxes[new_key] = {
                "bbox": bbox, "label": label,
                "color": color, "material": material, "description": description_text
            }

            already_seen = False

            # ======== CHECK FOR TRACKING TRANSITION (EXPLORATION ONLY) ========
            if in_exploration:
                transition, obj, distance = self.check_tracking_transition(
                    label_base, color, material, description_embedding, bbox
                )
                
                if transition:
                    self.log_both('warn', f"🔴 [TRANSITION] Switching from EXPLORATION to TRACKING mode")
                    self.exploration_mode = False
                    self.tracking_step_counter = 1
                    tracking_activated = True
                    in_exploration = False
                    self.exploration_frame_counter = 0
                    
                    # Publish transition event
                    msg = Bool()
                    msg.data = True
                    self.tracking_activated_pub.publish(msg)
                    
                    # Update object and continue in tracking mode with new position
                    updated_obj, dist, iou = self.modify_existing_object(obj, bbox, description_embedding)
                    current_perception_objects.append(updated_obj)
                    objects_modified = True
                    already_seen = True
                    continue

                # Normal exploration: check for duplicate in same position
                for obj in wm.persistent_perceptions:
                    obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label
                    if not hasattr(obj, "embedding"):
                        obj.embedding = get_embedding(client, obj.description)
                    if obj.embedding is None:
                        continue
                    
                    similarity = lost_similarity(world2vec, label_base, obj_label_base, color, obj.color,
                                               material, obj.material, description_embedding, obj.embedding)
                    if similarity > SIM_THRESHOLD and obj.bbox is not None:
                        iou = compute_iou_3d(bbox, obj.bbox)
                        if iou >= EXPLORATION_IOU_THRESHOLD:
                            already_seen = True
                            current_perception_objects.append(obj)
                            obj.bbox = bbox
                            break

            # ======== TRACKING LOGIC ========
            else:
                best_match = None
                best_score = 0

                for obj in wm.persistent_perceptions:
                    if not hasattr(obj, "embedding"):
                        obj.embedding = get_embedding(client, obj.description)
                    if obj.embedding is None:
                        continue

                    obj_label_base = obj.label.split('#')[0] if '#' in obj.label else obj.label
                    similarity = lost_similarity(world2vec, label_base, obj_label_base, color, obj.color,
                                               material, obj.material, description_embedding, obj.embedding)

                    if similarity > SIM_THRESHOLD and similarity > best_score:
                        best_score = similarity
                        best_match = obj

                if best_match:
                    already_seen = True
                    updated_obj, distance, iou = self.modify_existing_object(best_match, bbox, description_embedding)
                    current_perception_objects.append(updated_obj)
                    objects_modified = True

            # ======== ADD NEW OBJECT ========
            if not already_seen:
                new_obj = self.add_new_object(label, bbox, description_text, color, material,
                                             description_embedding, in_exploration)
                current_perception_objects.append(new_obj)
                objects_modified = True

        # Check if exploration frame limit reached (without seeing object again)
        if in_exploration and self.exploration_frame_counter >= EXPLORATION_FRAME_LIMIT and len(request.descriptions.descriptions) > 0:
            self.log_both('warn', f"🔴 [TRANSITION] Exploration frame limit ({EXPLORATION_FRAME_LIMIT}) reached - switching to TRACKING mode")
            self.exploration_mode = False
            self.tracking_step_counter = 1
            tracking_activated = True
            in_exploration = False
            self.exploration_frame_counter = 0
            
            # Publish transition event
            msg = Bool()
            msg.data = True
            self.tracking_activated_pub.publish(msg)

        # ======== DELETION LOGIC (TRACKING ONLY) ========
        if not in_exploration:
            description_received = len(request.descriptions.descriptions) > 0
            pov_volume = self.latest_fov_volume

            if pov_volume:
                objects_modified_delete = self.delete_undetected_objects(pov_volume, current_perception_objects, description_received)
                objects_modified = objects_modified or objects_modified_delete

                objects_modified_uncertain = self.delete_uncertain_objects(pov_volume)
                objects_modified = objects_modified or objects_modified_uncertain

        # ======== PUBLISH & SAVE ========
        if objects_modified and not in_exploration:
            publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
            publish_persistent_centroids(self, wm, self.persistent_centroids_pub)
            publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
            publish_uncertain_centroids(self, self.uncertain_objects, self.uncertain_centroids_pub)
            save_uncertain_objects(self)

        if not in_exploration:
            save_scene_graph(self, self.tracking_step_counter)

        self.latest_bboxes.clear()
        self.latest_fov_volume = None

        # Prepare response
        response.status = "tracking_activated" if tracking_activated else "success"
        response.num_objects = len(wm.persistent_perceptions)
        response.tracking_mode_activated = tracking_activated

        return response

    def add_new_object(self, label, bbox, description_text, color, material, description_embedding, in_exploration):
        """Add a new detected object to persistent_perceptions."""
        new_obj = Object(label, None, bbox, description_text, color, material)
        new_obj.embedding = description_embedding
        wm.persistent_perceptions.append(new_obj)

        phase = "exploration" if in_exploration else "tracking"
        step = self.exploration_step_counter if in_exploration else self.tracking_step_counter
        self.db.on_new_object(new_obj, phase=phase, step=step)

        x_size = bbox["x_max"] - bbox["x_min"]
        y_size = bbox["y_max"] - bbox["y_min"]
        z_size = bbox["z_max"] - bbox["z_min"]
        volume = x_size * y_size * z_size

        mode_tag = "[EXPLORATION]" if in_exploration else f"[TRACKING STEP {self.tracking_step_counter}]"
        self.log_both('info', f"{mode_tag} New object '{label}' (volume: {volume:.3f} m³)")

        if not in_exploration:
            tracking_logger.log_new_object(new_obj, case_type="NEW DETECTION")

        save_persistent_perceptions(self)

        if in_exploration:
            self.exploration_step_counter += 1
            save_scene_graph(self, self.exploration_step_counter, is_exploration=True)

        return new_obj

    def modify_existing_object(self, best_match, bbox, description_embedding):
        """Modify an existing tracked object."""
        old_bbox = best_match.bbox
        iou = compute_iou_3d(bbox, old_bbox)

        old_x = (old_bbox['x_min'] + old_bbox['x_max']) / 2.0
        old_y = (old_bbox['y_min'] + old_bbox['y_max']) / 2.0
        old_z = (old_bbox['z_min'] + old_bbox['z_max']) / 2.0
        new_x = (bbox['x_min'] + bbox['x_max']) / 2.0
        new_y = (bbox['y_min'] + bbox['y_max']) / 2.0
        new_z = (bbox['z_min'] + bbox['z_max']) / 2.0
        distance = np.sqrt((new_x - old_x)**2 + (new_y - old_y)**2 + (new_z - old_z)**2)

        if iou >= TRACKING_IOU_THRESHOLD:
            best_match.bbox = bbox
            return best_match, distance, iou
        else:
            if best_match in wm.persistent_perceptions:
                wm.persistent_perceptions.remove(best_match)
                self.db.on_object_moved(best_match, old_bbox=best_match.bbox, new_bbox=bbox,
                                       distance=distance, iou=iou, step=self.tracking_step_counter)

            if distance > 0.8:
                if best_match not in self.uncertain_objects:
                    self.uncertain_objects.append(best_match)
                    self.db.on_uncertain_added(best_match, step=self.tracking_step_counter)

            updated_obj = Object(best_match.label, None, bbox, best_match.description,
                               best_match.color, best_match.material)
            updated_obj.embedding = description_embedding
            wm.persistent_perceptions.append(updated_obj)

            tracking_logger.log_position_change(best_match.label, old_bbox, bbox, distance,
                                              step_number=self.tracking_step_counter, obj=updated_obj,
                                              case_type="POSITION UPDATE")

            return updated_obj, distance, iou

    def delete_undetected_objects(self, pov_volume, current_perception_objects, description_received):
        """Delete objects not detected in current frame within POV."""
        if not pov_volume:
            self.log_both('info', f"No FOV volume available from depth camera - unable to calculate POV")
            return False

        self.log_both('info', f"[POV] Using FOV from depth camera: X[{pov_volume['x_min']:.2f}, {pov_volume['x_max']:.2f}], "
              f"Y[{pov_volume['y_min']:.2f}, {pov_volume['y_max']:.2f}], "
              f"Z[{pov_volume['z_min']:.2f}, {pov_volume['z_max']:.2f}]")
        publish_pov_volume(self, pov_volume, self.considered_volume_pub)

        objects_to_remove = []

        if not description_received:
            for obj in wm.persistent_perceptions:
                if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                    objects_to_remove.append(obj)
                    self.log_both('info', f"DELETION: '{obj.label}' is IN POV but NOT SEEN → Will be REMOVED")
        else:
            for obj in wm.persistent_perceptions:
                if obj not in current_perception_objects:
                    if obj.bbox and bbox_centroid_in_volume(obj.bbox, pov_volume):
                        objects_to_remove.append(obj)

        if objects_to_remove:
            self.log_both('info', f"\nDELETING {len(objects_to_remove)} OBJECTS - TRACKING STEP {self.tracking_step_counter}")
            for obj in objects_to_remove:
                tracking_logger.log_deletion(obj.label, "Object in POV but not detected", bbox=obj.bbox,
                                            step_number=self.tracking_step_counter, obj=obj, case_type="NOT SEEN IN POV")
                wm.persistent_perceptions.remove(obj)
                self.db.on_object_deleted(obj, reason="not seen in POV",
                                  step=self.tracking_step_counter)

            save_persistent_perceptions(self)
            return True
        else:
            self.log_both('info', f"✓ No objects to remove in this step")
            return False

    def delete_uncertain_objects(self, pov_volume):
        """Remove uncertain objects whose zones have been verified in POV."""
        self.log_both('info', f"\n{'─'*60}")
        self.log_both('info', f"[TRACKING STEP {self.tracking_step_counter}] Verifying uncertain objects with POV VOLUME...")
        self.log_both('info', f"{'─'*60}")

        uncertain_to_remove = []

        if pov_volume:
            for uncertain_obj in self.uncertain_objects:
                if uncertain_obj.bbox and bbox_centroid_in_volume(uncertain_obj.bbox, pov_volume):
                    uncertain_to_remove.append(uncertain_obj)
                    self.log_both('info', f"✅ DELETION: '{uncertain_obj.label}' (ORANGE) in POV - zone VERIFIED → REMOVE")

        if uncertain_to_remove:
            self.log_both('info', f"\nDELETING {len(uncertain_to_remove)} UNCERTAIN OBJECTS - TRACKING STEP {self.tracking_step_counter}")
            for uncertain_obj in uncertain_to_remove:
                tracking_logger.log_deletion(uncertain_obj.label, "Uncertain zone verified", bbox=uncertain_obj.bbox,
                                            step_number=self.tracking_step_counter, obj=uncertain_obj,
                                            case_type="UNCERTAIN ZONE VERIFIED")
                self.uncertain_objects.remove(uncertain_obj)
            return True
        else:
            if self.uncertain_objects:
                self.log_both('info', f"No uncertain object to remove in this step")
            return False

    def periodic_bbox_publisher(self):
        """Periodically publishes persistent and uncertain bounding boxes."""
        if not self.exploration_mode:
            if len(wm.persistent_perceptions) > 0:
                publish_persistent_bboxes(self, wm, self.persistent_bbox_pub)
                publish_persistent_centroids(self, wm, self.persistent_centroids_pub)

            if len(self.uncertain_objects) > 0:
                publish_uncertain_bboxes(self, self.uncertain_objects, self.uncertain_bboxes_pub)
                publish_uncertain_centroids(self, self.uncertain_objects, self.uncertain_centroids_pub)

    def _descriptions_callback(self, msg):
        self.latest_descriptions = msg
        self._try_process()

    def _bboxes_callback(self, msg):
        self.latest_bboxes = msg
        self._try_process()

    def _try_process(self):
        """Processa solo quando ha sia descriptions che bboxes."""
        if self.latest_descriptions is None or self.latest_bboxes is None:
            return

        descriptions = self.latest_descriptions
        bboxes = self.latest_bboxes

        # Reset per il prossimo frame
        self.latest_descriptions = None
        self.latest_bboxes = None

        # Crea un request fittizio e chiama direttamente la logica
        request = ObjectTrackingService.Request()
        request.descriptions = descriptions
        request.bboxes = bboxes

        response = ObjectTrackingService.Response()
        self.object_tracking_callback(request, response)

        self.log_both('info', f'[TOPIC TRIGGER] status={response.status}, objects={response.num_objects}')

def main(args=None):
    rclpy.init(args=args)
    service_node = ObjectManagerService()

    try:
        rclpy.spin(service_node)
    except KeyboardInterrupt:
        print(f"OBJECT MANAGER SERVICE closed {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    finally:
        service_node.destroy_node()
        rclpy.shutdown()
        tracking_logger.close()


if __name__ == "__main__":
    main()