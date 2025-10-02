from manim import *
import manim

print(manim.__version__)
class Presentation(Scene):
    def construct(self):
        tile = Text("Support Vector Machines", font_size=72).center()
        self.play(FadeIn(tile))
        self.wait(2)
        self.play(FadeOut(tile))
        definition_SVM = Tex(
            "Support Vector Machine (SVM) is a supervised machine learning that use the regularization term and the Hinge Loss to increase the generalization ability while correctly classify the samples. It also use kernel trick to learn complex pattern.",
            font_size=35,
            tex_to_color_map={"Support Vector Machine (SVM)": YELLOW, "supervised": BLUE, "regularization": RED, "Hinge Loss": GREEN, "kernel trick": PURPLE}
        ).scale(0.9).move_to(ORIGIN)
        # "They aim to find the optimal hyperplane \n that separates different classes in the feature space to increase the generalization. \n It also uses kernel methods to handle non-linear data by mapping it to higher dimensions.", font_size=20).center()
        self.play(FadeIn(definition_SVM))
        self.wait(5)
        self.play(FadeOut(definition_SVM))
        obj_function = MathTex(r"\min_{w, b} \;", r" \frac{1}{2} \lVert w \rVert^2", r" +", r" C\ ",r"\sum_{i=1}^{n} \max \left( 0, \; 1 - y_i \left( w^T x_i + b \right) \right)", font_size=40).center()
        framboxReg = SurroundingRectangle(obj_function[1], color=RED, buff=0.1)
        capReg = Tex("Regularization Term", font_size=24, color=RED).next_to(framboxReg, UP)
        framboxHinge = SurroundingRectangle(obj_function[4], color=GREEN, buff=0.1)
        capHinge = Tex("Hinge Loss", font_size=24, color=GREEN).next_to(framboxHinge, UP)
        framboxC = SurroundingRectangle(obj_function[3], color=BLUE, buff=0.1)
        capC = Tex("Penalty Parameter", font_size=24, color=BLUE).next_to(framboxC, DOWN*2.5)
        self.play(FadeIn(obj_function))
        self.play(Create(framboxReg), Create(framboxHinge), Create(framboxC))
        self.play(FadeIn(capReg), FadeIn(capHinge), FadeIn(capC))
        self.wait(5)
        self.play(FadeOut(obj_function), FadeOut(framboxReg), FadeOut(framboxHinge), FadeOut(framboxC), FadeOut(capReg), FadeOut(capHinge), FadeOut(capC))



        #scene 2
        primal_no_constraint = Text("Primal without constraint", font_size= 36, color= YELLOW)
        primal_with_constraint = Text("Primal with constraint", font_size= 36, color= BLUE)
        #write primal equation with slack constraint
        primal_equation = MathTex(r"\min_{w, b, \xi} \; \frac{1}{2} \lVert w \rVert^2 + C \sum_{i=1}^{n} \xi_i\quad \text{s.t.} \quad y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0", font_size=36)
        primal_equation.next_to(primal_with_constraint, DOWN, buff=0.2)
        dual_form = Text("Dual form", font_size= 36, color= PURPLE)
        dual_equation = MathTex(
                r"\max_{\alpha} \; \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j \, ",r"x_i^T x_j",r"\quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \;\sum_{i=1}^{m} \alpha_i y_i = 0",
                font_size=36
            )
        w_eq_alpha_x = MathTex(
            r"\text{Let } w = \alpha X\text{, taking the derivative and multiply with the Lagrange multiplier}",
            font_size=36,
            tex_to_color_map={"w = \alpha X": RED}
        )

    

        # Arrange them vertically
        group = VGroup(
            primal_no_constraint,
            primal_with_constraint,
            primal_equation,
            w_eq_alpha_x,
            dual_form,
        ).arrange(DOWN, buff=0.8)
        dual_equation.next_to(group, DOWN, buff=0.2)
        # Create arrows
        arrows = VGroup(
            Arrow(primal_no_constraint.get_bottom(), primal_with_constraint.get_top(), buff=0.1),
            Arrow(primal_equation.get_bottom(), w_eq_alpha_x.get_top(), buff=0.1),
            Arrow(w_eq_alpha_x.get_bottom(), dual_form.get_top(), buff=0.1)
        )

        # Draw everything
        self.play(Write(group))
        self.play(Write(dual_equation))
        self.play(Create(arrows))
        self.wait(2)
        
        self.wait(2)
        self.play(FadeOut(group), FadeOut(arrows))

        #move dual equation to center
        self.play(dual_equation.animate.move_to(ORIGIN).scale(1.2))
        xTxBox = SurroundingRectangle(dual_equation[1], color=YELLOW, buff=0.1)
        self.play(Create(xTxBox))
        self.wait(2)
        self.play(FadeOut(xTxBox))

        xmaptophix = MathTex(r"x_i \mapsto \phi(x_i)", font_size=36, tex_to_color_map={r"\phi(x_i)": PURPLE}).next_to(dual_equation, DOWN, buff=1)
        self.play(Write(xmaptophix))
        self.wait(2)
        self.play(FadeOut(xmaptophix))
        #Transform dual equation to the phi space (Transform x_i^T x_j to phi(x_i)T phi(x_j))
        # Create a new MathTex object with the updated equation for phi(x_i)^T phi(x_j)
        dual_equation_phi = MathTex(
            r"\max_{\alpha} \; \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j \, \phi(x_i)^{\top} \phi(x_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \;\sum_{i=1}^{m} \alpha_i y_i = 0",
            font_size=36,
            tex_to_color_map={r"\phi(x_i)": PURPLE, r"\phi(x_j)": PURPLE}
        ).move_to(dual_equation.get_center())
        self.play(Transform(dual_equation, dual_equation_phi))
        self.wait(2)
        kernel = MathTex(r"K_{ij} = K(x_i, x_j) = \phi(x_i)^{\top} \phi(x_j)").next_to(dual_equation, DOWN, buff=1)
        self.play(Write(kernel))
        self.wait(5)
        # Transform phi(x_i)^T phi(x_j) to K(x_i, x_j)
        dual_equation_kernel = MathTex(
            r"\max_{\alpha} \; \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j \, K(x_i, x_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \;\sum_{i=1}^{m} \alpha_i y_i = 0",
            font_size=36,
            tex_to_color_map={r"K(x_i, x_j)": YELLOW}
        ).move_to(dual_equation.get_center())
        self.play(Transform(dual_equation, dual_equation_kernel))
        self.wait(5)
        self.play(FadeOut(dual_equation), FadeOut(kernel))

        self.wait(1)
        adv_title = Text("Advantages of SVM", font_size=48, color=GREEN)
        adv_list = BulletedList(
            "Avoid overfitting by default",
            "Effective in high dimensional space",
            "Rich hypothesis space / Complex pattern",
            font_size=36,
            color=WHITE
        ).next_to(adv_title, DOWN, buff=0.5)
        lim_title = Text("Limitations of SVM", font_size=48, color=RED)
        lim_list = BulletedList(
            "Pre-defined kernels",
            "Takes up memory spaces",
            "No probabilistic output",
            font_size=36,
            color=WHITE
        ).next_to(lim_title, DOWN, buff=0.5)
        # Group advantages and limitations side by side
        slide_group = VGroup(
            VGroup(adv_title, adv_list).arrange(DOWN, buff=0.3),
            VGroup(lim_title, lim_list).arrange(DOWN, buff=0.3)
        ).arrange(DOWN, buff=1).move_to(ORIGIN)
        self.play(FadeIn(slide_group))
        self.wait(4)
        self.play(FadeOut(slide_group))
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
        self.clear()
        # Turn into a normal scene
        self.wait(1)
        # Set to normal front view
        self.set_camera_orientation(phi=0 * DEGREES, theta=-90 * DEGREES)
        self.move_camera(zoom=1)
        self.wait(1)



        # STRART PRESENTATION





        tile = Text("Support Vector Machines", font_size=72).center()
        self.play(FadeIn(tile))
        self.wait(2)
        self.play(FadeOut(tile))
        definition_SVM = Tex(
            "Support Vector Machine (SVM) is a supervised machine learning that use the regularization term and the Hinge Loss to increase the generalization ability while correctly classify the samples. It also use kernel trick to learn complex pattern.",
            font_size=35,
            tex_to_color_map={"Support Vector Machine (SVM)": YELLOW, "supervised": BLUE, "regularization": RED, "Hinge Loss": GREEN, "kernel trick": PURPLE}
        ).scale(0.9).move_to(ORIGIN)
        # "They aim to find the optimal hyperplane \n that separates different classes in the feature space to increase the generalization. \n It also uses kernel methods to handle non-linear data by mapping it to higher dimensions.", font_size=20).center()
        self.play(FadeIn(definition_SVM))
        self.wait(5)
        self.play(FadeOut(definition_SVM))
        obj_function = MathTex(r"\min_{w, b} \;", r" \frac{1}{2} \lVert w \rVert^2", r" +", r" C\ ",r"\sum_{i=1}^{n} \max \left( 0, \; 1 - y_i \left( w^T x_i + b \right) \right)", font_size=40).center()
        framboxReg = SurroundingRectangle(obj_function[1], color=RED, buff=0.1)
        capReg = Tex("Regularization Term", font_size=24, color=RED).next_to(framboxReg, UP)
        framboxHinge = SurroundingRectangle(obj_function[4], color=GREEN, buff=0.1)
        capHinge = Tex("Hinge Loss", font_size=24, color=GREEN).next_to(framboxHinge, UP)
        framboxC = SurroundingRectangle(obj_function[3], color=BLUE, buff=0.1)
        capC = Tex("Penalty Parameter", font_size=24, color=BLUE).next_to(framboxC, DOWN*2.5)
        self.play(FadeIn(obj_function))
        self.play(Create(framboxReg), Create(framboxHinge), Create(framboxC))
        self.play(FadeIn(capReg), FadeIn(capHinge), FadeIn(capC))
        self.wait(5)
        self.play(FadeOut(obj_function), FadeOut(framboxReg), FadeOut(framboxHinge), FadeOut(framboxC), FadeOut(capReg), FadeOut(capHinge), FadeOut(capC))



        #scene 2
        primal_no_constraint = Text("Primal without constraint", font_size= 36, color= YELLOW)
        primal_with_constraint = Text("Primal with constraint", font_size= 36, color= BLUE)
        #write primal equation with slack constraint
        primal_equation = MathTex(r"\min_{w, b, \xi} \; \frac{1}{2} \lVert w \rVert^2 + C \sum_{i=1}^{n} \xi_i\quad \text{s.t.} \quad y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0", font_size=36)
        primal_equation.next_to(primal_with_constraint, DOWN, buff=0.2)
        dual_form = Text("Dual form", font_size= 36, color= PURPLE)
        dual_equation = MathTex(
                r"\max_{\alpha} \; \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j \, ",r"x_i^T x_j",r"\quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \;\sum_{i=1}^{m} \alpha_i y_i = 0",
                font_size=36
            )
        w_eq_alpha_x = MathTex(
            r"\text{Let } w = \alpha X\text{, taking the derivative and multiply with the Lagrange multiplier}",
            font_size=36,
            tex_to_color_map={"w = \alpha X": RED}
        )

    

        # Arrange them vertically
        group = VGroup(
            primal_no_constraint,
            primal_with_constraint,
            primal_equation,
            w_eq_alpha_x,
            dual_form,
        ).arrange(DOWN, buff=0.8)
        dual_equation.next_to(group, DOWN, buff=0.2)
        # Create arrows
        arrows = VGroup(
            Arrow(primal_no_constraint.get_bottom(), primal_with_constraint.get_top(), buff=0.1),
            Arrow(primal_equation.get_bottom(), w_eq_alpha_x.get_top(), buff=0.1),
            Arrow(w_eq_alpha_x.get_bottom(), dual_form.get_top(), buff=0.1)
        )

        # Draw everything
        self.play(Write(group))
        self.play(Write(dual_equation))
        self.play(Create(arrows))
        self.wait(2)
        
        self.wait(2)
        self.play(FadeOut(group), FadeOut(arrows))

        #move dual equation to center
        self.play(dual_equation.animate.move_to(ORIGIN).scale(1.2))
        xTxBox = SurroundingRectangle(dual_equation[1], color=YELLOW, buff=0.1)
        self.play(Create(xTxBox))
        self.wait(2)
        self.play(FadeOut(xTxBox))

        xmaptophix = MathTex(r"x_i \mapsto \phi(x_i)", font_size=36, tex_to_color_map={r"\phi(x_i)": PURPLE}).next_to(dual_equation, DOWN, buff=1)
        self.play(Write(xmaptophix))
        self.wait(2)
        self.play(FadeOut(xmaptophix))
        #Transform dual equation to the phi space (Transform x_i^T x_j to phi(x_i)T phi(x_j))
        # Create a new MathTex object with the updated equation for phi(x_i)^T phi(x_j)
        dual_equation_phi = MathTex(
            r"\max_{\alpha} \; \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j \, \phi(x_i)^{\top} \phi(x_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \;\sum_{i=1}^{m} \alpha_i y_i = 0",
            font_size=36,
            tex_to_color_map={r"\phi(x_i)": PURPLE, r"\phi(x_j)": PURPLE}
        ).move_to(dual_equation.get_center())
        self.play(Transform(dual_equation, dual_equation_phi))
        self.wait(2)
        kernel = MathTex(r"K_{ij} = K(x_i, x_j) = \phi(x_i)^{\top} \phi(x_j)").next_to(dual_equation, DOWN, buff=1)
        self.play(Write(kernel))
        self.wait(5)
        # Transform phi(x_i)^T phi(x_j) to K(x_i, x_j)
        dual_equation_kernel = MathTex(
            r"\max_{\alpha} \; \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j \, K(x_i, x_j) \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \;\sum_{i=1}^{m} \alpha_i y_i = 0",
            font_size=36,
            tex_to_color_map={r"K(x_i, x_j)": YELLOW}
        ).move_to(dual_equation.get_center())
        self.play(Transform(dual_equation, dual_equation_kernel))
        self.wait(5)
        self.play(FadeOut(dual_equation), FadeOut(kernel))

        self.wait(1)
        adv_title = Text("Advantages of SVM", font_size=48, color=GREEN)
        adv_list = BulletedList(
            "Avoid overfitting by default",
            "Effective in high dimensional space",
            "Rich hypothesis space / Complex pattern",
            font_size=36,
            color=WHITE
        ).next_to(adv_title, DOWN, buff=0.5)
        lim_title = Text("Limitations of SVM", font_size=48, color=RED)
        lim_list = BulletedList(
            "Pre-defined kernels",
            "Takes up memory spaces",
            "No probabilistic output",
            font_size=36,
            color=WHITE
        ).next_to(lim_title, DOWN, buff=0.5)
        # Group advantages and limitations side by side
        slide_group = VGroup(
            VGroup(adv_title, adv_list).arrange(DOWN, buff=0.3),
            VGroup(lim_title, lim_list).arrange(DOWN, buff=0.3)
        ).arrange(DOWN, buff=1).move_to(ORIGIN)
        self.play(FadeIn(slide_group))
        self.wait(4)
        self.play(FadeOut(slide_group))
