from manim import *

class VisualizeSVM(ThreeDScene):
    def construct(self):
        # Set up the 3D axes
        axes = ThreeDAxes(x_range=[-5, 5, 1], y_range=[-5, 5, 1], z_range=[-5, 5, 1])
        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        points = [
            (1, 1, 0), (-1, -1, 0), (1, -1, 0), (-1, 1, 0)
        ]
        surface = Surface(
            lambda u, v: np.array([u, v, 0]),
            u_range=[-5, 5],
            v_range=[-5, 5],
            fill_opacity=0.1,
            color=BLUE_E,
            stroke_width=0
        )
        colors = [ RED_A , RED_A, BLUE, BLUE]
        data_points = VGroup()
        for i in range(4):
            print(colors[i])
            point = Dot3D(radius=0.05, color=colors[i]).move_to(points[i])
            data_points.add(point)
        self.add(axes, data_points, surface)
        
        # Create a XOR problem with 4 data points on a 2D grid
        self.begin_ambient_camera_rotation(rate=0.1)
        


        # map the points to a higher dimension using a polynomial kernel
        phi = lambda x, y: np.array([x, y, x * y])
        transformed_points = [phi(*point[:2]) for point in points]
        transformed_data_points = VGroup()
        for i in range(4):
            point = Dot3D(radius=0.05, color=colors[i]).move_to(transformed_points[i])
            transformed_data_points.add(point)
        #transform the surface respect to the kernel
        transformed_surface = Surface(
            lambda u, v: np.array([u, v, u * v]),
            u_range=[-5, 5],
            v_range=[-5, 5],
            fill_opacity=0.1,
            color=BLUE_E,
            stroke_width=0
        )
        self.wait(2)

        hyperplane = Surface(
            lambda u, v: np.array([u, v, 0]),
            u_range=[-5, 5],
            v_range=[-5, 5],
            fill_opacity=0.1,
            checkerboard_colors= [RED_A,RED_A],
            stroke_width=0
        )

        self.play(Transform(data_points, transformed_data_points),Transform(surface, transformed_surface), run_time=3)
        self.wait(2)
        self.play(FadeIn(hyperplane), run_time=1)
        self.wait(5)
        self.stop_ambient_camera_rotation()
        self.wait(0.4)
