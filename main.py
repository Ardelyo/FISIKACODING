import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from collections import deque
import colorsys

# --- Inisialisasi Pygame ---
pygame.init()

# Ukuran jendela
WIDTH, HEIGHT = 1280, 720
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Sandbox Fisika 2D")

# Warna
LIGHT_GRAY = (200, 200, 200)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0) # Warna baru untuk tombol Play

# --- Inisialisasi Pymunk ---
space = pymunk.Space()
space.gravity = (0, -981)  # Gravitasi ke bawah

# Opsi debug menggambar Pymunk (opsional, untuk visualisasi lebih detail)
draw_options = pymunk.pygame_util.DrawOptions(screen)

# --- Barriers Statis ---
def create_barrier(space, p1, p2, thickness=5):
    barrier_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    barrier_shape = pymunk.Segment(barrier_body, p1, p2, thickness)
    barrier_shape.friction = 0.9
    barrier_shape.elasticity = 0.8
    space.add(barrier_body, barrier_shape)
    return barrier_shape

# Membuat lantai dan dinding
floor_shape = create_barrier(space, (0, 50), (WIDTH, 50))  # Lantai
left_wall = create_barrier(space, (5, 50), (5, HEIGHT))    # Dinding kiri
right_wall = create_barrier(space, (WIDTH-5, 50), (WIDTH-5, HEIGHT))  # Dinding kanan
ceiling = create_barrier(space, (0, HEIGHT-5), (WIDTH, HEIGHT-5))     # Langit-langit

# --- Fungsi untuk Membuat Lingkaran ---
def create_circle(space, position, mass, friction, elasticity):
    """Membuat body dinamis dengan bentuk lingkaran dan menambahkannya ke ruang fisika."""
    radius = 20
    inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
    body = pymunk.Body(mass, inertia)
    body.position = position

    shape = pymunk.Circle(body, radius)
    shape.friction = friction
    shape.elasticity = elasticity
    space.add(body, shape)
    return shape

# --- Fungsi untuk Membuat Kotak ---
def create_box(space, position, mass, friction, elasticity):
    """Membuat body dinamis dengan bentuk kotak dan menambahkannya ke ruang fisika."""
    size = 40
    inertia = pymunk.moment_for_box(mass, (size, size))
    body = pymunk.Body(mass, inertia)
    body.position = position

    shape = pymunk.Poly.create_box(body, (size, size))
    shape.friction = friction
    shape.elasticity = elasticity
    space.add(body, shape)
    return shape

# --- Fungsi untuk Menggambar Objek Pymunk ---
def draw_objects(screen, space):
    """Menggambar semua bentuk (shapes) dari ruang fisika ke layar Pygame."""
    # Draw grid
    grid_spacing = 50
    for x in range(0, WIDTH - UI_PANEL_WIDTH, grid_spacing):
        pygame.draw.line(screen, (220, 220, 220), (x, 0), (x, HEIGHT), 1)
    for y in range(0, HEIGHT, grid_spacing):
        pygame.draw.line(screen, (220, 220, 220), (0, y), (WIDTH - UI_PANEL_WIDTH, y), 1)

    for shape in space.shapes:
        if isinstance(shape, pymunk.Circle):
            p = shape.body.position + shape.offset.rotated(shape.body.angle)
            
            # Color based on velocity
            if shape.body.body_type == pymunk.Body.DYNAMIC:
                color = get_color_by_velocity(shape.body.velocity)
                # Fill circle
                pygame.draw.circle(screen, color, (int(p.x), int(HEIGHT - p.y)), int(shape.radius))
                # Draw outline
                pygame.draw.circle(screen, BLACK, (int(p.x), int(HEIGHT - p.y)), int(shape.radius), 2)
                
                # Draw velocity vector
                if shape.body.velocity.length > 0.1:
                    draw_vector(screen, p, shape.body.velocity, (255, 0, 0))
                
        elif isinstance(shape, pymunk.Segment):
            # Menggambar segmen (lantai dan barriers)
            p1 = shape.body.position + shape.a.rotated(shape.body.angle)
            p2 = shape.body.position + shape.b.rotated(shape.body.angle)
            pygame.draw.line(screen, BLACK, (int(p1.x), int(HEIGHT - p1.y)),
                           (int(p2.x), int(HEIGHT - p2.y)), int(shape.radius * 2))
            
        elif isinstance(shape, pymunk.Poly):
            if len(shape.vertices) == 4:  # Kotak
                vertices = []
                for v in shape.vertices:
                    p = shape.body.position + v.rotated(shape.body.angle)
                    vertices.append((int(p.x), int(HEIGHT - p.y)))
                
                # Color based on velocity
                if shape.body.body_type == pymunk.Body.DYNAMIC:
                    color = get_color_by_velocity(shape.body.velocity)
                    pygame.draw.polygon(screen, color, vertices)  # Fill
                    pygame.draw.polygon(screen, BLACK, vertices, 2)  # Outline
                    
                    # Draw velocity vector from center
                    center = shape.body.position
                    if shape.body.velocity.length > 0.1:
                        draw_vector(screen, center, shape.body.velocity, (255, 0, 0))
                else:
                    pygame.draw.polygon(screen, BLACK, vertices, 2)

# --- Data Collection dan Visualisasi ---
class DataCollector:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.ke_data = deque(maxlen=max_points)
        self.pe_data = deque(maxlen=max_points)
        self.total_energy_data = deque(maxlen=max_points)
        self.velocity_x_data = deque(maxlen=max_points)
        self.velocity_y_data = deque(maxlen=max_points)
        self.time_points = deque(maxlen=max_points)
        self.start_time = pygame.time.get_ticks()

    def update(self, body):
        if body is None:
            return
        
        current_time = (pygame.time.get_ticks() - self.start_time) / 1000.0
        ke = 0.5 * body.mass * body.velocity.length_squared
        pe = body.mass * abs(space.gravity.y) * (body.position.y - 50)
        
        self.ke_data.append(ke)
        self.pe_data.append(pe)
        self.total_energy_data.append(ke + pe)
        self.velocity_x_data.append(body.velocity.x)
        self.velocity_y_data.append(body.velocity.y)
        self.time_points.append(current_time)

def draw_graph(screen, data, rect, color, max_val, min_val, label):
    """Draw a graph in the specified rectangle."""
    if not data:
        return

    pygame.draw.rect(screen, WHITE, rect, 2)
    
    # Draw axes
    pygame.draw.line(screen, BLACK, (rect.left + 5, rect.bottom - 5),
                    (rect.right - 5, rect.bottom - 5), 1)  # X axis
    pygame.draw.line(screen, BLACK, (rect.left + 5, rect.top + 5),
                    (rect.left + 5, rect.bottom - 5), 1)  # Y axis

    # Plot data
    points = []
    for i, value in enumerate(data):
        x = rect.left + 5 + (i / len(data)) * (rect.width - 10)
        normalized_value = (value - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        y = rect.bottom - 5 - normalized_value * (rect.height - 10)
        points.append((x, y))

    if len(points) > 1:
        pygame.draw.lines(screen, color, False, points, 2)

    # Draw label
    label_surface = FONT.render(label, True, BLACK)
    screen.blit(label_surface, (rect.left + 5, rect.top + 5))

def get_color_by_velocity(velocity):
    """Return a color based on velocity magnitude."""
    speed = velocity.length
    max_speed = 1000  # Adjust this value based on your simulation
    hue = min(speed / max_speed, 1.0)  # Map speed to hue (0-1)
    rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return tuple(int(x * 255) for x in rgb)

def draw_vector(screen, start_pos, vector, color, scale=0.5):
    """Draw a vector with an arrow head."""
    if vector.length < 0.1:  # Don't draw very small vectors
        return
    
    end_pos = start_pos + vector * scale
    pygame.draw.line(screen, color, 
                    (int(start_pos.x), int(HEIGHT - start_pos.y)),
                    (int(end_pos.x), int(HEIGHT - end_pos.y)), 2)
    
    # Draw arrow head
    angle = vector.angle
    arrow_length = 10
    arrow1 = pymunk.Vec2d(arrow_length, 0).rotated(angle + 2.8)
    arrow2 = pymunk.Vec2d(arrow_length, 0).rotated(angle - 2.8)
    pygame.draw.line(screen, color,
                    (int(end_pos.x), int(HEIGHT - end_pos.y)),
                    (int(end_pos.x - arrow1.x), int(HEIGHT - (end_pos.y - arrow1.y))), 2)
    pygame.draw.line(screen, color,
                    (int(end_pos.x), int(HEIGHT - end_pos.y)),
                    (int(end_pos.x - arrow2.x), int(HEIGHT - (end_pos.y - arrow2.y))), 2)

# --- UI Setup ---
UI_PANEL_WIDTH = 400  # Increased width for graphs
UI_PANEL_X = WIDTH - UI_PANEL_WIDTH
FONT = pygame.font.Font(None, 24)

# Initialize data collector
data_collector = DataCollector()

class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.label = label
        self.grabbed = False

    def draw(self, screen, font):
        pygame.draw.rect(screen, WHITE, self.rect, 2) # Border
        # Draw slider bar
        pygame.draw.line(screen, BLACK, (self.rect.x + 5, self.rect.centery), (self.rect.right - 5, self.rect.centery), 2)
        # Draw slider knob
        knob_x = self.rect.x + 5 + (self.val - self.min_val) / (self.max_val - self.min_val) * (self.rect.width - 10)
        pygame.draw.circle(screen, BLUE, (int(knob_x), self.rect.centery), 8)

        # Draw label and value
        text_label = font.render(f"{self.label}: {self.val:.2f}", True, BLACK)
        screen.blit(text_label, (self.rect.x, self.rect.y - 20))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.grabbed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.grabbed = False
        elif event.type == pygame.MOUSEMOTION:
            if self.grabbed:
                # Update value based on mouse position
                mouse_x, _ = event.pos
                normalized_x = (mouse_x - (self.rect.x + 5)) / (self.rect.width - 10)
                self.val = self.min_val + normalized_x * (self.max_val - self.min_val)
                self.val = max(self.min_val, min(self.max_val, self.val)) # Clamp value

# Tombol
button_circle_rect = pygame.Rect(UI_PANEL_X + 20, 50, 160, 40)
button_box_rect = pygame.Rect(UI_PANEL_X + 20, 100, 160, 40)

# Slider Properti Objek
slider_mass = Slider(UI_PANEL_X + 20, 200, 160, 20, 1, 100, 10, "Massa")
slider_friction = Slider(UI_PANEL_X + 20, 250, 160, 20, 0, 1, 0.7, "Gesekan")
slider_elasticity = Slider(UI_PANEL_X + 20, 300, 160, 20, 0, 1, 0.9, "Elastisitas")

object_sliders = [slider_mass, slider_friction, slider_elasticity]

# Slider Gravitasi
slider_gravity = Slider(UI_PANEL_X + 20, 400, 160, 20, -2000, 2000, -981, "Gravitasi Y")

# Tombol Kontrol Simulasi
button_play_rect = pygame.Rect(UI_PANEL_X + 20, 450, 75, 40)
button_pause_rect = pygame.Rect(UI_PANEL_X + 105, 450, 75, 40)
button_reset_rect = pygame.Rect(UI_PANEL_X + 20, 500, 160, 40)
button_clear_all_rect = pygame.Rect(UI_PANEL_X + 20, 550, 160, 40)

# Variabel untuk melacak bentuk yang dipilih
selected_shape_type = "circle" # Default

# Variabel untuk interaksi drag
dragged_body = None
dragged_joint = None
mouse_body = pymunk.Body(body_type=pymunk.Body.STATIC) # Body statis untuk mouse

# Kontrol Simulasi
simulation_running = True

# Objek yang sedang dipilih untuk menampilkan data
selected_object_body = None

# --- Fungsi untuk Menampilkan Data Objek ---
def display_object_data(screen, font, body, y_start_pos):
    """Menampilkan data fisika dari body yang dipilih."""
    if body is None:
        return

    # Posisi
    pos_text = font.render(f"Posisi: ({body.position.x:.1f}, {HEIGHT - body.position.y:.1f})", True, BLACK)
    screen.blit(pos_text, (UI_PANEL_X + 10, y_start_pos))

    # Kecepatan
    vel_text = font.render(f"Kecepatan: ({body.velocity.x:.1f}, {-body.velocity.y:.1f})", True, BLACK)
    screen.blit(vel_text, (UI_PANEL_X + 10, y_start_pos + 20))

    # Energi Kinetik
    ke = 0.5 * body.mass * body.velocity.length_squared
    ke_text = font.render(f"Energi Kinetik: {ke:.2f}", True, BLACK)
    screen.blit(ke_text, (UI_PANEL_X + 10, y_start_pos + 40))

    # Energi Potensial (relatif terhadap lantai y=50)
    # Perhatikan bahwa gravitasi Pymunk adalah negatif untuk ke bawah
    # Jadi, PE = m * g * h, di mana h adalah ketinggian dari lantai
    # Ketinggian = body.position.y - 50 (karena lantai di y=50)
    pe = body.mass * abs(space.gravity.y) * (body.position.y - 50)
    pe_text = font.render(f"Energi Potensial: {pe:.2f}", True, BLACK)
    screen.blit(pe_text, (UI_PANEL_X + 10, y_start_pos + 60))

    # Energi Total
    total_energy = ke + pe
    total_energy_text = font.render(f"Energi Total: {total_energy:.2f}", True, BLACK)
    screen.blit(total_energy_text, (UI_PANEL_X + 10, y_start_pos + 80))


# --- Loop Utama Aplikasi ---
running = True
clock = pygame.time.Clock()
FPS = 60

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            pymunk_y = HEIGHT - mouse_y
            mouse_pos_pymunk = (mouse_x, pymunk_y)

            # Cek apakah klik di area UI
            if mouse_x > UI_PANEL_X:
                if button_circle_rect.collidepoint(mouse_x, mouse_y):
                    selected_shape_type = "circle"
                    selected_object_body = None # Deselect object when changing creation type
                elif button_box_rect.collidepoint(mouse_x, mouse_y):
                    selected_shape_type = "box"
                    selected_object_body = None # Deselect object when changing creation type
                elif button_play_rect.collidepoint(mouse_x, mouse_y):
                    simulation_running = True
                elif button_pause_rect.collidepoint(mouse_x, mouse_y):
                    simulation_running = False
                elif button_reset_rect.collidepoint(mouse_x, mouse_y):
                    # Reset simulasi: hapus semua objek dinamis dan inisialisasi ulang space
                    for shape in space.shapes[:]: # Iterasi salinan list agar aman saat menghapus
                        if shape.body.body_type == pymunk.Body.DYNAMIC:
                            space.remove(shape, shape.body)
                    simulation_running = True # Mulai lagi setelah reset
                    selected_object_body = None # Deselect object on reset
                elif button_clear_all_rect.collidepoint(mouse_x, mouse_y):
                    for shape in space.shapes[:]:
                        if shape.body.body_type == pymunk.Body.DYNAMIC:
                            space.remove(shape, shape.body)
                    selected_object_body = None # Deselect object on clear all

                # Handle slider events
                for slider in object_sliders + [slider_gravity]:
                    slider.handle_event(event)
            else:
                # Coba ambil objek yang ada untuk drag atau seleksi
                hit_shape = space.point_query_nearest(mouse_pos_pymunk, 0, pymunk.ShapeFilter())
                if hit_shape and hit_shape.shape.body.body_type == pymunk.Body.DYNAMIC:
                    dragged_body = hit_shape.shape.body
                    selected_object_body = dragged_body # Select the dragged object
                    # Buat joint antara mouse_body dan objek yang di-drag
                    dragged_joint = pymunk.PivotJoint(mouse_body, dragged_body, mouse_pos_pymunk)
                    dragged_joint.max_force = 100000 # Batasi kekuatan agar tidak terlalu kaku
                    space.add(dragged_joint)
                else:
                    # Buat objek baru jika tidak ada objek yang di-drag atau diklik
                    selected_object_body = None # Deselect if clicking empty space
                    current_mass = slider_mass.val
                    current_friction = slider_friction.val
                    current_elasticity = slider_elasticity.val

                    if selected_shape_type == "circle":
                        create_circle(space, (mouse_x, pymunk_y), current_mass, current_friction, current_elasticity)
                    elif selected_shape_type == "box":
                        create_box(space, (mouse_x, pymunk_y), current_mass, current_friction, current_elasticity)
        elif event.type == pygame.MOUSEBUTTONUP:
            # Lepaskan objek yang di-drag
            if dragged_joint:
                # Dapatkan kecepatan mouse saat ini
                mouse_rel_x, mouse_rel_y = pygame.mouse.get_rel()
                impulse_vector = pymunk.Vec2d(mouse_rel_x * 50, -mouse_rel_y * 50) # Faktor pengali untuk kekuatan lemparan
                
                if dragged_body:
                    dragged_body.apply_impulse_at_local_point(impulse_vector, (0, 0))

                space.remove(dragged_joint)
                dragged_joint = None
                dragged_body = None
            for slider in object_sliders + [slider_gravity]:
                slider.handle_event(event)
        elif event.type == pygame.MOUSEMOTION:
            mouse_x, mouse_y = event.pos
            pymunk_y = HEIGHT - mouse_y
            mouse_pos_pymunk = (mouse_x, pymunk_y)

            # Perbarui posisi mouse_body jika ada objek yang di-drag
            if dragged_body:
                mouse_body.position = mouse_pos_pymunk
            for slider in object_sliders + [slider_gravity]:
                slider.handle_event(event)

    # Update gravitasi berdasarkan slider
    space.gravity = (0, slider_gravity.val)

    # Update ruang fisika hanya jika simulasi berjalan
    if simulation_running:
        space.step(1 / FPS)

    # Gambar latar belakang
    screen.fill(LIGHT_GRAY)

    # Gambar objek fisika
    draw_objects(screen, space)

    # Gambar panel UI
    pygame.draw.rect(screen, (180, 180, 180), (UI_PANEL_X, 0, UI_PANEL_WIDTH, HEIGHT))
    pygame.draw.line(screen, BLACK, (UI_PANEL_X, 0), (UI_PANEL_X, HEIGHT), 2)

    # Gambar tombol pilihan bentuk
    pygame.draw.rect(screen, (100, 100, 100) if selected_shape_type == "circle" else (150, 150, 150), button_circle_rect)
    pygame.draw.rect(screen, (100, 100, 100) if selected_shape_type == "box" else (150, 150, 150), button_box_rect)

    # Teks tombol pilihan bentuk
    text_circle = FONT.render("Lingkaran", True, BLACK)
    text_box = FONT.render("Kotak", True, BLACK)
    screen.blit(text_circle, (button_circle_rect.x + (button_circle_rect.width - text_circle.get_width()) // 2, button_circle_rect.y + (button_circle_rect.height - text_circle.get_height()) // 2))
    screen.blit(text_box, (button_box_rect.x + (button_box_rect.width - text_box.get_width()) // 2, button_box_rect.y + (button_box_rect.height - text_box.get_height()) // 2))

    # Gambar slider properti objek
    for slider in object_sliders:
        slider.draw(screen, FONT)

    # Gambar slider gravitasi
    slider_gravity.draw(screen, FONT)

    # Gambar tombol kontrol simulasi
    pygame.draw.rect(screen, GREEN if simulation_running else (150, 150, 150), button_play_rect)
    pygame.draw.rect(screen, RED if not simulation_running else (150, 150, 150), button_pause_rect)
    pygame.draw.rect(screen, (150, 150, 150), button_reset_rect)
    pygame.draw.rect(screen, (150, 150, 150), button_clear_all_rect)

    # Teks tombol kontrol simulasi
    text_play = FONT.render("Play", True, BLACK)
    text_pause = FONT.render("Pause", True, BLACK)
    text_reset = FONT.render("Reset", True, BLACK)
    text_clear_all = FONT.render("Hapus Semua", True, BLACK)

    screen.blit(text_play, (button_play_rect.x + (button_play_rect.width - text_play.get_width()) // 2, button_play_rect.y + (button_play_rect.height - text_play.get_height()) // 2))
    screen.blit(text_pause, (button_pause_rect.x + (button_pause_rect.width - text_pause.get_width()) // 2, button_pause_rect.y + (button_pause_rect.height - text_pause.get_height()) // 2))
    screen.blit(text_reset, (button_reset_rect.x + (button_reset_rect.width - text_reset.get_width()) // 2, button_reset_rect.y + (button_reset_rect.height - text_reset.get_height()) // 2))
    screen.blit(text_clear_all, (button_clear_all_rect.x + (button_clear_all_rect.width - text_clear_all.get_width()) // 2, button_clear_all_rect.y + (button_clear_all_rect.height - text_clear_all.get_height()) // 2))

    # Tampilkan data objek yang dipilih
    display_object_data(screen, FONT, selected_object_body, 600) # Mulai di y=600 untuk data objek

    # Perbarui tampilan
    pygame.display.flip()

    # Batasi FPS
    clock.tick(FPS)

pygame.quit()