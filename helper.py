from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    Parser,
)

def CartTriplePendulumPlant(
        m_cart=1, # kg
        m1=1, # kg
        m2=1, # kg
        m3=1, # kg
        l1=1, # m
        l2=1, # m
        l3=1, # m
):
    sdf_string = f"""
    <?xml version="1.0" ?>
    <sdf version="1.7">
      <model name="cart_triple_pendulum">

        <link name="cart">
          <pose>0 0 0 0 0 0</pose>
          <inertial>
            <mass>{m_cart}</mass>
          </inertial>
          <visual name="cart_body">
            <geometry>
              <box>
                <size>0.5 0.3 0.3</size>
              </box>
            </geometry>
            <material>
              <diffuse>0.5 0.5 0.5 1</diffuse>
            </material>
          </visual>
          <visual name="cart_wheel1">
            <pose>-0.15 0 -0.2 1.57 0 0</pose>
            <geometry>
              <cylinder>
                <radius>0.05</radius>
                <length>0.3</length>
              </cylinder>
            </geometry>
            <material>
              <diffuse>0 0 0 1</diffuse>
            </material>
          </visual>
          <visual name="cart_wheel2">
            <pose>0.15 0 -0.2 1.57 0 0</pose>
            <geometry>
              <cylinder>
                <radius>0.05</radius>
                <length>0.3</length>
              </cylinder>
            </geometry>
            <material>
              <diffuse>0 0 0 1</diffuse>
            </material>
          </visual>
        </link>

        <link name="pendulum1">
          <pose>0 -0.17 0 0 0 0</pose>
          <inertial>
            <pose>0 0 {-l1/2} 0 0 0</pose>
            <mass>{m1}</mass>
            <inertia>
              <ixx>{m1 * l1**2 / 12}</ixx>
              <iyy>{m1 * l1**2 / 12}</iyy>
              <izz>1e-5</izz>
            </inertia>
          </inertial>
          <visual name="pendulum1_rod">
            <pose>0 0 {-l1/2} 0 0 0</pose>
            <geometry>
              <cylinder>
                <radius>0.02</radius>
                <length>{l1}</length>
              </cylinder>
            </geometry>
            <material>
              <diffuse>1 0 0 1</diffuse>
            </material>
          </visual>
        </link>

        <link name="pendulum2">
          <pose>0 -0.21 {-l1} 0 0 0</pose>
          <inertial>
            <pose>0 0 {-l2/2} 0 0 0</pose>
            <mass>{m2}</mass>
            <inertia>
              <ixx>{m2 * l2**2 / 12}</ixx>
              <iyy>{m2 * l2**2 / 12}</iyy>
              <izz>1e-5</izz>
            </inertia>
          </inertial>
          <visual name="pendulum2_rod">
            <pose>0 0 {-l2/2} 0 0 0</pose>
            <geometry>
              <cylinder>
                <radius>0.02</radius>
                <length>{l2}</length>
              </cylinder>
            </geometry>
            <material>
              <diffuse>0 1 0 1</diffuse>
            </material>
          </visual>
        </link>

        <link name="pendulum3">
          <pose>0 -0.25 {-l1-l2} 0 0 0</pose>
          <inertial>
            <pose>0 0 {-l3/2} 0 0 0</pose>
            <mass>{m3}</mass>
            <inertia>
              <ixx>{m3 * l3**2 / 12}</ixx>
              <iyy>{m3 * l3**2 / 12}</iyy>
              <izz>1e-5</izz>
            </inertia>
          </inertial>
          <visual name="pendulum3_rod">
            <pose>0 0 {-l3/2} 0 0 0</pose>
            <geometry>
              <cylinder>
                <radius>0.02</radius>
                <length>{l3}</length>
              </cylinder>
            </geometry>
            <material>
              <diffuse>0 0 1 1</diffuse>
            </material>
          </visual>
        </link>

        <joint name="linear" type="prismatic">
          <parent>world</parent>
          <child>cart</child>
          <axis>
            <xyz>1 0 0</xyz>
          </axis>
        </joint>

        <joint name="joint1" type="continuous">
          <parent>cart</parent>
          <child>pendulum1</child>
          <axis>
            <xyz>0 -1 0</xyz>
            <limit>
              <effort>0</effort>
            </limit>
          </axis>
        </joint>

        <joint name="joint2" type="continuous">
          <parent>pendulum1</parent>
          <child>pendulum2</child>
          <axis>
            <xyz>0 -1 0</xyz>
            <limit>
              <effort>0</effort>
            </limit>
          </axis>
        </joint>

        <joint name="joint3" type="continuous">
          <parent>pendulum2</parent>
          <child>pendulum3</child>
          <axis>
            <xyz>0 -1 0</xyz>
            <limit>
              <effort>0</effort>
            </limit>
          </axis>
        </joint>

        <link name="ground">
          <visual name="ground_visual">
            <pose>0 0 -0.3 0 0 0</pose>
            <geometry>
              <box>
                <size>50 0.3 0.1</size>
              </box>
            </geometry>
            <material>
              <diffuse>0.9 0.9 0.9 1</diffuse>
            </material>
          </visual>
        </link>
        <joint name="ground_fixed" type="fixed">
          <parent>world</parent>
          <child>ground</child>
        </joint>

      </model>
    </sdf>
    """

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant, scene_graph).AddModelsFromString(sdf_string, "sdf")
    plant.Finalize()

    builder.ExportInput(plant.get_actuation_input_port(), "f_cart")
    builder.ExportOutput(plant.get_state_output_port(), "x")
    builder.ExportOutput(scene_graph.get_query_output_port(), "query")

    cart_triple_pendulum = builder.Build()
    return cart_triple_pendulum