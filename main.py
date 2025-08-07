import pygame
import pymunk
import math
import json
import csv
from collections import deque
import time

# --- Inisialisasi Utama ---
pygame.init()

# --- Konfigurasi & Konstanta ---
WIDTH, HEIGHT = 1600, 900
FPS = 60

# Warna
COLOR_BG = (30, 30, 40)
COLOR_GRID = (50, 50, 60)
UI_BG = (60, 65, 75)
UI_TEXT = (220, 220, 220)
UI_BORDER = (40, 45, 55)
UI_BUTTON = (80, 85, 95)
UI_BUTTON_HOVER = (100, 105, 115)
UI_BUTTON_SELECTED = (65, 130, 210)
CYAN_HIGHLIGHT = (0, 255, 255)
RED = (220, 50, 50)
GREEN = (50, 200, 50)
BLUE = (50, 100, 220)

# UI
UI_PANEL_WIDTH = 350
UI_PANEL_X = WIDTH - UI_PANEL_WIDTH

# Font
FONT_NORMAL = pygame.font.Font(None, 24)
FONT_TITLE = pygame.font.Font(None, 28)
FONT_HEADER = pygame.font.Font(None, 36)

# --- Kelas Kamera ---
class Camera:
    def __init__(self):
        self.offset = pygame.math.Vector2(0, 0)
        self.zoom = 1.0

    def world_to_screen(self, world_pos):
        return (int((world_pos[0] - self.offset.x) * self.zoom + WIDTH / 2),
                int((world_pos[1] - self.offset.y) * self.zoom + HEIGHT / 2))

    def screen_to_world(self, screen_pos):
        return (int((screen_pos[0] - WIDTH / 2) / self.zoom + self.offset.x),
                int((screen_pos[1] - HEIGHT / 2) / self.zoom + self.offset.y))

    def zoom_at(self, amount, focus_point):
        old_zoom = self.zoom
        self.zoom *= amount
        self.zoom = max(0.1, min(self.zoom, 5.0))
        self.offset += (focus_point - self.offset) * (1 - self.zoom / old_zoom)

# --- Kelas UI ---
class Button:
    def __init__(self, rect, text, font, callback=None):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.font = font
        self.callback = callback
        self.is_hovered = False

    def draw(self, screen, is_selected=False):
        color = UI_BUTTON_SELECTED if is_selected else (UI_BUTTON_HOVER if self.is_hovered else UI_BUTTON)
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, UI_BORDER, self.rect, 1, border_radius=5)
        text_surf = self.font.render(self.text, True, UI_TEXT)
        screen.blit(text_surf, text_surf.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.is_hovered:
            if self.callback:
                self.callback()
            return True
        return False

class Slider:
    def __init__(self, rect, label, min_val, max_val, initial_val):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.min_val, self.max_val, self.val = min_val, max_val, initial_val
        self.grabbed = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.grabbed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.grabbed = False
        elif event.type == pygame.MOUSEMOTION and self.grabbed:
            ratio = (event.pos[0] - self.rect.x) / self.rect.width
            self.val = self.min_val + ratio * (self.max_val - self.min_val)
            self.val = max(self.min_val, min(self.max_val, self.val))

    def draw(self, screen):
        text_surf = FONT_NORMAL.render(f"{self.label}: {self.val:.2f}", True, UI_TEXT)
        screen.blit(text_surf, (self.rect.x, self.rect.y - 22))
        pygame.draw.rect(screen, UI_BORDER, self.rect, border_radius=5)
        ratio = (self.val - self.min_val) / (self.max_val - self.min_val)
        fill_width = int(self.rect.width * ratio)
        pygame.draw.rect(screen, UI_BUTTON_SELECTED, (self.rect.x, self.rect.y, fill_width, self.rect.height), border_radius=5)

# --- Kelas Pengumpul Data & Grafik ---
class DataCollector:
    def __init__(self, max_points=200):
        self.max_points = max_points
        self.data = {}
        self.labels = {
            'pos_x': 'Position X', 'pos_y': 'Position Y',
            'vel_x': 'Velocity X', 'vel_y': 'Velocity Y', 'vel_mag': 'Speed',
            'ke': 'Kinetic E', 'pe': 'Potential E', 'total_e': 'Total E'
        }
        self.reset()

    def reset(self):
        for key in self.labels:
            self.data[key] = deque(maxlen=self.max_points)

    def update(self, body, gravity_y):
        if body is None or body.mass == 0 or body.body_type != pymunk.Body.DYNAMIC: return
        pos = body.position
        vel = body.velocity
        ke = 0.5 * body.mass * vel.length_squared
        pe = -body.mass * gravity_y * (pos.y - HEIGHT/2) # PE relatif
        
        self.data['pos_x'].append(pos.x)
        self.data['pos_y'].append(pos.y)
        self.data['vel_x'].append(vel.x)
        self.data['vel_y'].append(vel.y)
        self.data['vel_mag'].append(vel.length)
        self.data['ke'].append(ke)
        self.data['pe'].append(pe)
        self.data['total_e'].append(ke + pe)
        
def draw_graph(screen, rect, data_deque, label, color):
    if not data_deque: return
    pygame.draw.rect(screen, UI_BG, rect)
    pygame.draw.rect(screen, UI_BORDER, rect, 1)

    max_val = max(data_deque) if data_deque else 1.0
    min_val = min(data_deque) if data_deque else 0.0
    val_range = max(1e-5, max_val - min_val)
    
    points = []
    for i, val in enumerate(data_deque):
        x = rect.x + (i / (len(data_deque) - 1 if len(data_deque) > 1 else 1)) * rect.width
        y = rect.bottom - ((val - min_val) / val_range) * rect.height
        points.append((x, y))

    if len(points) > 1: pygame.draw.lines(screen, color, False, points, 2)
    label_surf = FONT_NORMAL.render(f"{label}: {data_deque[-1]:.1f}", True, UI_TEXT)
    screen.blit(label_surf, (rect.x + 5, rect.y + 5))

# --- Fungsi Utama ---
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Physics Sandbox Pro")
    clock = pygame.time.Clock()
    
    # --- State & Variabel ---
    space = pymunk.Space()
    space.gravity = (0, -981)
    
    camera = Camera()
    
    # Koleksi Objek
    objects = []
    joints = []
    
    # State UI
    simulation_running = True
    current_tab = "Tools"
    current_tool = "SELECT"
    
    # State Interaksi
    selected_body = None
    dragged_body = None
    dragged_joint = None
    mouse_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    
    # State Alat
    polygon_points = []
    joint_tool_body1 = None

    # State Statistik
    data_collector = DataCollector()
    path_tracer = deque(maxlen=200)
    current_plot_var = 'ke'
    next_body_id = 0

    # Tambahkan batas statis
    static_lines = [
        pymunk.Segment(space.static_body, (0, 0), (WIDTH, 0), 5),
        pymunk.Segment(space.static_body, (0, 0), (0, HEIGHT), 5),
        pymunk.Segment(space.static_body, (WIDTH, 0), (WIDTH, HEIGHT), 5),
        pymunk.Segment(space.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 5),
    ]
    for line in static_lines:
        line.elasticity = 0.9
        line.friction = 0.7
    space.add(*static_lines)


    # --- Fungsi Bantuan ---
    def get_body_by_id(body_id):
        for obj in objects:
            if obj.get('id') == body_id:
                return obj['shape'].body
        return None

    def remove_object(body_to_remove):
        nonlocal objects, joints, selected_body
        
        # Hapus objek dan bentuknya
        obj_to_remove = None
        for obj in objects:
            if obj['shape'].body == body_to_remove:
                obj_to_remove = obj
                break
        if obj_to_remove:
            space.remove(obj_to_remove['shape'], obj_to_remove['shape'].body)
            objects.remove(obj_to_remove)
            if selected_body == body_to_remove:
                selected_body = None
                data_collector.reset()
                path_tracer.clear()

        # Hapus sambungan yang terhubung
        joints_to_remove = [j for j in joints if j['constraint'].a == body_to_remove or j['constraint'].b == body_to_remove]
        for joint in joints_to_remove:
            space.remove(joint['constraint'])
            joints.remove(joint)

    def set_current_tool(tool_name):
        nonlocal current_tool, polygon_points, joint_tool_body1
        current_tool = tool_name
        polygon_points = []
        joint_tool_body1 = None
        print(f"Tool changed to: {tool_name}")

    def set_current_tab(tab_name):
        nonlocal current_tab
        current_tab = tab_name
    
    def set_plot_var(var_name):
        nonlocal current_plot_var
        current_plot_var = var_name

    def export_csv():
        if not selected_body: 
            print("No object selected to export data.")
            return
        filename = f"stats_export_{int(time.time())}.csv"
        header = list(data_collector.data.keys())
        rows = zip(*[data_collector.data[key] for key in header])
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"Data exported to {filename}")


    # --- UI Elements ---
    sliders = {
        'density': Slider(pygame.Rect(UI_PANEL_X + 20, 280, UI_PANEL_WIDTH - 40, 10), "Density", 0.1, 5.0, 1.0),
        'friction': Slider(pygame.Rect(UI_PANEL_X + 20, 330, UI_PANEL_WIDTH - 40, 10), "Friction", 0.0, 2.0, 0.7),
        'elasticity': Slider(pygame.Rect(UI_PANEL_X + 20, 380, UI_PANEL_WIDTH - 40, 10), "Elasticity", 0.0, 1.5, 0.8),
        'gravity_y': Slider(pygame.Rect(UI_PANEL_X + 20, 100, UI_PANEL_WIDTH - 40, 10), "Gravity Y", -2000, 2000, -981),
        'damping': Slider(pygame.Rect(UI_PANEL_X + 20, 150, UI_PANEL_WIDTH - 40, 10), "Air Damping", 0.9, 1.0, 0.998)
    }

    # Tombol Tab
    tab_buttons = [
        Button(pygame.Rect(UI_PANEL_X + 0, 20, UI_PANEL_WIDTH/3, 40), "Tools", FONT_NORMAL, lambda: set_current_tab("Tools")),
        Button(pygame.Rect(UI_PANEL_X + UI_PANEL_WIDTH/3, 20, UI_PANEL_WIDTH/3, 40), "World", FONT_NORMAL, lambda: set_current_tab("World")),
        Button(pygame.Rect(UI_PANEL_X + 2*UI_PANEL_WIDTH/3, 20, UI_PANEL_WIDTH/3, 40), "Statistics", FONT_NORMAL, lambda: set_current_tab("Statistics"))
    ]
    
    # Tombol Alat
    tool_names = ["SELECT", "CIRCLE", "BOX", "POLYGON", "PIN_JOINT", "SPRING", "ERASER"]
    tool_buttons = []
    for i, name in enumerate(tool_names):
        rect = pygame.Rect(UI_PANEL_X + 20 + (i % 3) * 105, 100 + (i // 3) * 50, 100, 40)
        tool_buttons.append(Button(rect, name, FONT_NORMAL, lambda n=name: set_current_tool(n)))
        
    # Tombol Save/Load/Clear
    scene_buttons = [
        Button(pygame.Rect(UI_PANEL_X + 20, HEIGHT - 110, 100, 40), "Save", FONT_NORMAL, lambda: save_scene()),
        Button(pygame.Rect(UI_PANEL_X + 125, HEIGHT - 110, 100, 40), "Load", FONT_NORMAL, lambda: load_scene()),
        Button(pygame.Rect(UI_PANEL_X + 230, HEIGHT - 110, 100, 40), "Clear", FONT_NORMAL, lambda: clear_scene())
    ]
    
    # Tombol Statistik
    plot_vars = list(data_collector.labels.keys())
    stat_buttons = [Button(pygame.Rect(UI_PANEL_X + 20, HEIGHT - 60, 310, 40), "Export CSV", FONT_NORMAL, export_csv)]
    for i, var in enumerate(plot_vars):
        rect = pygame.Rect(UI_PANEL_X + 20 + (i % 4) * 80, 250 + (i//4)*40, 75, 35)
        stat_buttons.append(Button(rect, data_collector.labels[var], FONT_NORMAL, lambda v=var: set_plot_var(v)))

    all_ui_elements = tab_buttons + tool_buttons + scene_buttons + stat_buttons
    
    def clear_scene():
        nonlocal selected_body, dragged_body, dragged_joint, joint_tool_body1, next_body_id
        selected_body = dragged_body = dragged_joint = joint_tool_body1 = None
        polygon_points.clear()
        
        for j in list(space.constraints): space.remove(j)
        for s in list(space.shapes):
            if s.body.body_type == pymunk.Body.DYNAMIC:
                space.remove(s, s.body)

        objects.clear()
        joints.clear()
        data_collector.reset()
        path_tracer.clear()
        next_body_id = 0
        print("Scene cleared.")

    def save_scene(filename="scene.json"):
        nonlocal next_body_id
        scene_data = {
            'next_body_id': next_body_id,
            'camera': {'offset_x': camera.offset.x, 'offset_y': camera.offset.y, 'zoom': camera.zoom},
            'world': {'gravity_y': sliders['gravity_y'].val, 'damping': sliders['damping'].val},
            'objects': [],
            'joints': []
        }
        for obj in objects:
            body = obj['shape'].body
            obj_data = {
                'id': obj['id'],
                'type': obj['type'],
                'pos': (body.position.x, body.position.y),
                'angle': body.angle,
                'vel': (body.velocity.x, body.velocity.y),
                'ang_vel': body.angular_velocity,
                'friction': obj['shape'].friction,
                'elasticity': obj['shape'].elasticity
            }
            if obj['type'] == 'circle':
                obj_data['radius'] = obj['shape'].radius
                obj_data['mass'] = body.mass
            elif obj['type'] == 'box':
                obj_data['size'] = obj['size']
                obj_data['mass'] = body.mass
            elif obj['type'] == 'polygon':
                obj_data['vertices'] = obj['vertices']
                obj_data['mass'] = body.mass
            scene_data['objects'].append(obj_data)
        
        for joint in joints:
            c = joint['constraint']
            joint_data = {
                'type': joint['type'],
                'body_a_id': getattr(c.a, '_id', None),
                'body_b_id': getattr(c.b, '_id', None),
                'anchor_a': c.anchor_a,
                'anchor_b': c.anchor_b,
            }
            if joint['type'] == 'spring':
                joint_data['rest_length'] = c.rest_length
                joint_data['stiffness'] = c.stiffness
                joint_data['damping'] = c.damping
            scene_data['joints'].append(joint_data)

        with open(filename, 'w') as f:
            json.dump(scene_data, f, indent=2)
        print(f"Scene saved to {filename}")

    def load_scene(filename="scene.json"):
        nonlocal next_body_id
        clear_scene()
        try:
            with open(filename, 'r') as f:
                scene_data = json.load(f)
            
            next_body_id = scene_data.get('next_body_id', 0)
            cam_data = scene_data.get('camera', {})
            camera.offset.x = cam_data.get('offset_x', 0)
            camera.offset.y = cam_data.get('offset_y', 0)
            camera.zoom = cam_data.get('zoom', 1.0)

            world_data = scene_data.get('world', {})
            sliders['gravity_y'].val = world_data.get('gravity_y', -981)
            sliders['damping'].val = world_data.get('damping', 0.998)
            
            created_bodies = {}
            for obj_data in scene_data.get('objects', []):
                pos = tuple(obj_data['pos'])
                mass = obj_data.get('mass', 1)
                body = None

                if obj_data['type'] == 'circle':
                    radius = obj_data['radius']
                    moment = pymunk.moment_for_circle(mass, 0, radius)
                    body = pymunk.Body(mass, moment)
                    shape = pymunk.Circle(body, radius)
                    objects.append({'id': obj_data['id'], 'type': 'circle', 'shape': shape})
                elif obj_data['type'] == 'box':
                    size = tuple(obj_data['size'])
                    moment = pymunk.moment_for_box(mass, size)
                    body = pymunk.Body(mass, moment)
                    shape = pymunk.Poly.create_box(body, size)
                    objects.append({'id': obj_data['id'], 'type': 'box', 'size': size, 'shape': shape})
                elif obj_data['type'] == 'polygon':
                    vertices = [tuple(v) for v in obj_data['vertices']]
                    moment = pymunk.moment_for_poly(mass, vertices)
                    body = pymunk.Body(mass, moment)
                    shape = pymunk.Poly(body, vertices)
                    objects.append({'id': obj_data['id'], 'type': 'polygon', 'vertices': vertices, 'shape': shape})
                
                if body and shape:
                    body.position = pos
                    body.angle = obj_data['angle']
                    body.velocity = tuple(obj_data['vel'])
                    body.angular_velocity = obj_data['ang_vel']
                    shape.friction = obj_data['friction']
                    shape.elasticity = obj_data['elasticity']
                    body._id = obj_data['id']
                    created_bodies[body._id] = body
                    space.add(body, shape)

            for joint_data in scene_data.get('joints', []):
                body_a = created_bodies.get(joint_data['body_a_id'])
                body_b = created_bodies.get(joint_data['body_b_id'])
                if not body_a or not body_b: continue
                
                constraint = None
                if joint_data['type'] == 'pin':
                    constraint = pymunk.PinJoint(body_a, body_b, tuple(joint_data['anchor_a']), tuple(joint_data['anchor_b']))
                elif joint_data['type'] == 'spring':
                    constraint = pymunk.DampedSpring(body_a, body_b, tuple(joint_data['anchor_a']), tuple(joint_data['anchor_b']), 
                                                     joint_data['rest_length'], joint_data['stiffness'], joint_data['damping'])

                if constraint:
                    joints.append({'type': joint_data['type'], 'constraint': constraint})
                    space.add(constraint)

            print(f"Scene loaded from {filename}")
        except FileNotFoundError:
            print(f"Error: {filename} not found.")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error reading scene file: {e}")

    # --- Loop Utama ---
    running = True
    panning = False
    
    while running:
        mouse_pos = pygame.mouse.get_pos()
        world_mouse_pos = camera.screen_to_world(mouse_pos)
        
        is_mouse_on_ui = mouse_pos[0] > UI_PANEL_X
        
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Panning dengan tombol tengah mouse
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2 and not is_mouse_on_ui: panning = True
            if event.type == pygame.MOUSEBUTTONUP and event.button == 2: panning = False
            if event.type == pygame.MOUSEMOTION and panning:
                camera.offset.x -= event.rel[0] / camera.zoom
                camera.offset.y -= event.rel[1] / camera.zoom
            
            # Zoom dengan scroll wheel
            if event.type == pygame.MOUSEWHEEL and not is_mouse_on_ui:
                camera.zoom_at(1.1 if event.y > 0 else 0.9, pygame.math.Vector2(world_mouse_pos))

            # Event UI
            if is_mouse_on_ui:
                for elem in all_ui_elements: elem.handle_event(event)
                for s in sliders.values(): s.handle_event(event)
            # Event Dunia
            else: 
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Klik Kiri
                        hit = space.point_query_nearest(world_mouse_pos, 0, pymunk.ShapeFilter())
                        
                        if current_tool == "SELECT":
                            if hit and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                                body = hit.shape.body
                                if selected_body != body:
                                    data_collector.reset()
                                    path_tracer.clear()
                                selected_body = body
                                dragged_body = body
                                dragged_joint = pymunk.PivotJoint(mouse_body, dragged_body, world_mouse_pos)
                                dragged_joint.max_force = 50000 * body.mass
                                space.add(dragged_joint)
                        
                        elif current_tool == "ERASER":
                             if hit and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                                remove_object(hit.shape.body)
                        
                        elif current_tool == "CIRCLE":
                            radius = 40
                            mass = sliders['density'].val * math.pi * radius**2 / 1000
                            moment = pymunk.moment_for_circle(mass, 0, radius)
                            body = pymunk.Body(mass, moment)
                            body.position = world_mouse_pos
                            shape = pymunk.Circle(body, radius)
                            shape.friction = sliders['friction'].val
                            shape.elasticity = sliders['elasticity'].val
                            body._id = next_body_id; next_body_id += 1
                            space.add(body, shape)
                            objects.append({'id': body._id, 'type': 'circle', 'shape': shape})

                        elif current_tool == "BOX":
                            size = (80, 80)
                            mass = sliders['density'].val * size[0] * size[1] / 1000
                            moment = pymunk.moment_for_box(mass, size)
                            body = pymunk.Body(mass, moment)
                            body.position = world_mouse_pos
                            shape = pymunk.Poly.create_box(body, size)
                            shape.friction = sliders['friction'].val
                            shape.elasticity = sliders['elasticity'].val
                            body._id = next_body_id; next_body_id += 1
                            space.add(body, shape)
                            objects.append({'id': body._id, 'type': 'box', 'size': size, 'shape': shape})

                        elif current_tool == "POLYGON":
                            polygon_points.append(world_mouse_pos)

                        elif current_tool in ["PIN_JOINT", "SPRING"]:
                            if hit and hit.shape.body.body_type == pymunk.Body.DYNAMIC:
                                if not joint_tool_body1:
                                    joint_tool_body1 = hit.shape.body
                                else:
                                    body_a = joint_tool_body1
                                    body_b = hit.shape.body
                                    if body_a != body_b:
                                        anchor_a = body_a.world_to_local(body_a.position)
                                        anchor_b = body_b.world_to_local(world_mouse_pos)
                                        
                                        constraint = None
                                        if current_tool == "PIN_JOINT":
                                            constraint = pymunk.PinJoint(body_a, body_b, anchor_a, anchor_b)
                                            joints.append({'type': 'pin', 'constraint': constraint})
                                        elif current_tool == "SPRING":
                                            rest_length = body_a.position.get_distance(body_b.position)
                                            constraint = pymunk.DampedSpring(body_a, body_b, anchor_a, anchor_b, rest_length, 2000, 30)
                                            joints.append({'type': 'spring', 'constraint': constraint})
                                        
                                        if constraint:
                                            space.add(constraint)
                                    joint_tool_body1 = None # Reset alat
                            
                    if event.button == 3: # Klik Kanan
                        if current_tool == "POLYGON" and len(polygon_points) > 2:
                            mass = sliders['density'].val * 10
                            moment = pymunk.moment_for_poly(mass, polygon_points, (0,0))
                            body = pymunk.Body(mass, moment)
                            body.position = pymunk.vec2d.Vec2d(*pymunk.util.calc_center_of_gravity(polygon_points))
                            
                            local_verts = [body.world_to_local(p) for p in polygon_points]
                            shape = pymunk.Poly(body, local_verts)
                            shape.friction = sliders['friction'].val
                            shape.elasticity = sliders['elasticity'].val
                            body._id = next_body_id; next_body_id += 1
                            space.add(body, shape)
                            objects.append({'id': body._id, 'type': 'polygon', 'vertices': local_verts, 'shape': shape})
                            polygon_points.clear()
                        elif current_tool in ["PIN_JOINT", "SPRING", "POLYGON"]:
                            # Batalkan aksi
                            joint_tool_body1 = None
                            polygon_points.clear()

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1: # Lepas Klik Kiri
                        if dragged_joint:
                            space.remove(dragged_joint)
                            dragged_joint = None
                            dragged_body = None
                
                if event.type == pygame.MOUSEMOTION:
                    mouse_body.position = world_mouse_pos
        
        # --- Update Fisika ---
        if simulation_running:
            space.gravity = (0, sliders['gravity_y'].val)
            space.damping = sliders['damping'].val
            space.step(1 / FPS)
        
        # --- Update Data ---
        if selected_body and simulation_running:
            data_collector.update(selected_body, space.gravity.y)
            path_tracer.append(selected_body.position)

        # --- Gambar ---
        screen.fill(COLOR_BG)
        
        # Gambar Objek Fisika
        for obj in objects:
            body = obj['shape'].body
            color = RED if body == selected_body else BLUE
            
            if obj['type'] == 'circle':
                screen_pos = camera.world_to_screen(body.position)
                screen_radius = int(obj['shape'].radius * camera.zoom)
                if screen_pos[0] > -screen_radius and screen_pos[0] < WIDTH + screen_radius and \
                   screen_pos[1] > -screen_radius and screen_pos[1] < HEIGHT + screen_radius:
                    pygame.draw.circle(screen, color, screen_pos, screen_radius)
                    # Gambar garis sudut untuk menunjukkan rotasi
                    end_pos_local = pymunk.Vec2d(obj['shape'].radius, 0).rotated(body.angle)
                    end_pos_world = body.position + end_pos_local
                    end_pos_screen = camera.world_to_screen(end_pos_world)
                    pygame.draw.line(screen, UI_TEXT, screen_pos, end_pos_screen, 2)

            elif obj['type'] in ['box', 'polygon']:
                verts = [camera.world_to_screen(body.local_to_world(v)) for v in obj['shape'].get_vertices()]
                pygame.draw.polygon(screen, color, verts)
        
        # Gambar Sambungan
        for joint in joints:
            c = joint['constraint']
            a_pos = camera.world_to_screen(c.a.local_to_world(c.anchor_a))
            b_pos = camera.world_to_screen(c.b.local_to_world(c.anchor_b))
            pygame.draw.line(screen, CYAN_HIGHLIGHT, a_pos, b_pos, 2 if joint['type'] == 'pin' else 3)
            if joint['type'] == 'spring': # Gambar kumparan
                dir_vec = (pygame.math.Vector2(b_pos) - pygame.math.Vector2(a_pos))
                if dir_vec.length() > 0:
                    perp_vec = dir_vec.rotate(90).normalize() * 5
                    for i in range(1, 10):
                        point1 = pygame.math.Vector2(a_pos).lerp(b_pos, (i - 0.5) / 10) + perp_vec * (1 if i % 2 == 0 else -1)
                        point2 = pygame.math.Vector2(a_pos).lerp(b_pos, (i + 0.5) / 10) + perp_vec * (-1 if i % 2 == 0 else 1)
                        pygame.draw.line(screen, CYAN_HIGHLIGHT, point1, point2, 1)

        # Gambar Jejak Lintasan
        if len(path_tracer) > 1:
            screen_points = [camera.world_to_screen(p) for p in path_tracer]
            pygame.draw.lines(screen, GREEN, False, screen_points, 2)
            
        # Gambar Pratinjau Alat
        if current_tool == "POLYGON" and len(polygon_points) > 0:
            screen_points = [camera.world_to_screen(p) for p in polygon_points]
            if len(screen_points) > 1:
                pygame.draw.lines(screen, UI_TEXT, False, screen_points, 2)
            for p in screen_points:
                pygame.draw.circle(screen, UI_TEXT, p, 4)
        elif current_tool in ["PIN_JOINT", "SPRING"] and joint_tool_body1:
            start_pos = camera.world_to_screen(joint_tool_body1.position)
            end_pos = mouse_pos
            pygame.draw.line(screen, CYAN_HIGHLIGHT, start_pos, end_pos, 3, )
            pygame.draw.circle(screen, UI_BUTTON_SELECTED, start_pos, 8)
        
        # Gambar UI Panel
        pygame.draw.rect(screen, UI_BG, (UI_PANEL_X, 0, UI_PANEL_WIDTH, HEIGHT))
        pygame.draw.line(screen, UI_BORDER, (UI_PANEL_X, 0), (UI_PANEL_X, HEIGHT), 2)
        
        # Gambar Tombol Tab
        for btn in tab_buttons: btn.draw(screen, is_selected=(current_tab == btn.text))
        y_cursor = 70
        
        # Gambar Konten Tab
        if current_tab == "Tools":
            title_surf = FONT_TITLE.render("Tools", True, UI_TEXT)
            screen.blit(title_surf, (UI_PANEL_X + 20, y_cursor))
            y_cursor += 30
            for btn in tool_buttons: btn.draw(screen, is_selected=(current_tool == btn.text))
            
            y_cursor += 120
            title_surf = FONT_TITLE.render("Object Properties", True, UI_TEXT)
            screen.blit(title_surf, (UI_PANEL_X + 20, y_cursor))
            y_cursor += 40
            sliders['density'].rect.y = y_cursor; sliders['density'].draw(screen); y_cursor += 50
            sliders['friction'].rect.y = y_cursor; sliders['friction'].draw(screen); y_cursor += 50
            sliders['elasticity'].rect.y = y_cursor; sliders['elasticity'].draw(screen); y_cursor += 50
        elif current_tab == "World":
            title_surf = FONT_TITLE.render("World Properties", True, UI_TEXT)
            screen.blit(title_surf, (UI_PANEL_X + 20, y_cursor))
            y_cursor += 40
            sliders['gravity_y'].draw(screen)
            y_cursor += 50
            sliders['damping'].draw(screen)
        elif current_tab == "Statistics":
            title_surf = FONT_TITLE.render("Statistics", True, UI_TEXT)
            screen.blit(title_surf, (UI_PANEL_X + 20, y_cursor))
            y_cursor += 40
            
            if selected_body:
                draw_graph(screen, pygame.Rect(UI_PANEL_X + 20, y_cursor, 310, 100), 
                           data_collector.data[current_plot_var], 
                           data_collector.labels[current_plot_var], GREEN)
                y_cursor += 110
                
                plot_title_surf = FONT_NORMAL.render("Plot Variable:", True, UI_TEXT)
                screen.blit(plot_title_surf, (UI_PANEL_X + 20, y_cursor))
                y_cursor += 30
                
                for btn in stat_buttons:
                    if btn.callback == export_csv: continue # Lewati tombol export
                    is_selected = (current_plot_var == next(k for k,v in data_collector.labels.items() if v == btn.text))
                    btn.draw(screen, is_selected)
            else:
                text_surf = FONT_NORMAL.render("Select a dynamic object to see stats.", True, UI_TEXT)
                screen.blit(text_surf, (UI_PANEL_X + 20, y_cursor + 40))

        # Gambar Tombol Scene (selalu terlihat)
        for btn in scene_buttons: btn.draw(screen)
        # Gambar Tombol Statistik (selalu terlihat)
        for btn in stat_buttons:
             if btn.callback == export_csv:
                btn.draw(screen)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == '__main__':
    main()