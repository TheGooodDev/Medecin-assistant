# rotation_polygon_gravity_gif.py
import math, sys, pygame, imageio, numpy as np
from pygame.math import Vector2

# ----- Réglages gÉnÉraux -----
WIDTH, HEIGHT  = 700, 700
CENTER         = Vector2(WIDTH//2, HEIGHT//2)
BG_COLOR       = "black"

NUM_SIDES      = 6
POLY_RADIUS    = 250
ANGULAR_SPEED  = math.radians(40)
GRAVITY        = Vector2(0, 900)
BALL_RADIUS    = 18
ELASTICITY     = 0.9

DURATION_SEC   = 6          # longueur du GIF (≈ secondes simulées)
FPS            = 60         # images par seconde

# ----- Init -----
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock  = pygame.time.Clock()

ball_pos = Vector2(CENTER)
ball_vel = Vector2(120, -50)
angle    = 0.0

def regular_polygon(n, r, rot):
    return [Vector2(math.cos(2*math.pi*i/n+rot),
                    math.sin(2*math.pi*i/n+rot))*r + CENTER
            for i in range(n)]

def collide_ball(pos, vel, verts):
    for i in range(len(verts)):
        a, b = verts[i], verts[(i+1)%len(verts)]
        edge = b-a
        n_in = Vector2(-edge.y, edge.x).normalize()
        dist = (pos-a).dot(n_in)
        if dist < BALL_RADIUS:
            pos += (BALL_RADIUS-dist)*n_in
            v_n  = vel.dot(n_in)*n_in
            v_t  = vel - v_n
            vel  = v_t - v_n*ELASTICITY
    return pos, vel

frames = []  # ici seront stockées les images

# ----- Boucle principale -----
for _ in range(int(DURATION_SEC*FPS)):
    dt = clock.tick(FPS)/1000
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit(); sys.exit()

    angle   += ANGULAR_SPEED*dt
    verts    = regular_polygon(NUM_SIDES, POLY_RADIUS, angle)
    ball_vel += GRAVITY*dt
    ball_pos += ball_vel*dt
    ball_pos, ball_vel = collide_ball(ball_pos, ball_vel, verts)

    screen.fill(BG_COLOR)
    pygame.draw.polygon(screen, "white", verts, width=3)
    pygame.draw.circle(screen, "red", ball_pos, BALL_RADIUS)
    pygame.display.flip()

    # ----- Capture frame -----
    surf = pygame.surfarray.array3d(screen)       # (w,h,3)
    frame = np.transpose(surf, (1,0,2))           # (h,w,3) pour ImageIO
    frames.append(frame)

pygame.quit()

# ----- Écriture du GIF -----
imageio.mimsave("rotation_polygon.gif", frames, fps=FPS)
print("GIF enregistré → rotation_polygon.gif")
