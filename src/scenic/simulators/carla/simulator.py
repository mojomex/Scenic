"""Simulator interface for CARLA."""

try:
    import carla
except ImportError as e:
    raise ModuleNotFoundError('CARLA scenarios require the "carla" Python package') from e

import math
import os
import warnings

import scenic.core.errors as errors

if errors.verbosityLevel == 0:  # suppress pygame advertisement at zero verbosity
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"

import pygame

from scenic.core.simulators import SimulationCreationError
from scenic.domains.driving.simulators import DrivingSimulation, DrivingSimulator
from scenic.simulators.carla.blueprints import oldBlueprintNames
import scenic.simulators.carla.utils.utils as utils
import scenic.simulators.carla.utils.visuals as visuals
from scenic.syntax.veneer import verbosePrint
from scenic.simulators.carla.utils.data_recorder import DataRecorder


class CarlaSimulator(DrivingSimulator):
    """Implementation of `Simulator` for CARLA."""

    def __init__(
        self,
        carla_map,
        map_path,
        address="127.0.0.1",
        port=2000,
        timeout=5,
        render=True,
        record="",
        timestep=0.1,
        traffic_manager_port=None,
    ):
        super().__init__()
        verbosePrint(f"Connecting to CARLA on port {port}")
        self.client = carla.Client(address, port)
        self.client.set_timeout(20)  # increase timeout for world loading
        if carla_map is not None:
            try:
                self.world = self.client.load_world(carla_map)
            except Exception as e:
                raise RuntimeError(f"CARLA could not load world '{carla_map}'") from e
        else:
            if map_path.endswith(".xodr"):
                with open(map_path) as odr_file:
                    self.world = self.client.generate_opendrive_world(odr_file.read())
            else:
                raise RuntimeError("CARLA only supports OpenDrive maps")
        self.timestep = timestep

        self.client.set_timeout(timeout)  # decrease timeout for normal operation

        if traffic_manager_port is None:
            traffic_manager_port = port + 6000
        self.tm = self.client.get_trafficmanager(traffic_manager_port)
        self.tm.set_synchronous_mode(True)

        # Set to synchronous with fixed timestep
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = timestep  # NOTE: Should not exceed 0.1
        self.world.apply_settings(settings)
        verbosePrint("Map loaded in simulator.")

        self.render = render  # visualization mode ON/OFF
        self.record = record  # whether to use the carla recorder
        self.scenario_number = 0  # Number of the scenario executed

    def createSimulation(self, scene, *, timestep, **kwargs):
        if timestep is not None and timestep != self.timestep:
            raise RuntimeError(
                "cannot customize timestep for individual CARLA simulations; "
                "set timestep when creating the CarlaSimulator instead"
            )

        self.scenario_number += 1
        return CarlaSimulation(
            scene,
            self.client,
            self.tm,
            self.render,
            self.record,
            self.scenario_number,
            timestep=self.timestep,
            **kwargs,
        )

    def destroy(self):
        super().destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        self.tm.set_synchronous_mode(False)


class CarlaSimulation(DrivingSimulation):
    def __init__(self, scene, client, tm, render, record, scenario_number, **kwargs):
        self.client = client
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprintLib = self.world.get_blueprint_library()
        self.tm = tm
        self.render = render
        self.record = record
        self.scenario_number = scenario_number
        self.cameraManager = None
        self.semsegManager = None
        self.weather = str(scene.params.get("weather"))
        self.data_recorder = DataRecorder(weather=self.weather)
        self.frame_num = 0

        self.display: pygame.Surface = scene.display
        self.hud = visuals.HUD(*self.display.get_size() if self.display is not None else (8, 8))

        super().__init__(scene, **kwargs)

    def setup(self):
        weather = self.scene.params.get("weather")
        if weather is not None:
            if isinstance(weather, str):
                self.world.set_weather(getattr(carla.WeatherParameters, weather))
            elif isinstance(weather, dict):
                self.world.set_weather(carla.WeatherParameters(**weather))

        # Setup HUD
        if self.render:
            self.cameraManager = None
            self.semsegManager = None

        if self.record:
            if not os.path.exists(self.record):
                os.mkdir(self.record)

            name = "{}/scenario{}.log".format(self.record, self.scenario_number)
            self.client.start_recorder(name)
        
        # Despawn any cars that survived cleanup of the last simulation
        for actor in self.world.get_actors():
            if isinstance(actor, carla.Vehicle):
                actor.destroy()

        # Create objects.
        super().setup()

        # Set up camera manager and collision sensor for ego
        if self.render:
            egoActor = self.objects[0].carlaActor
            self.cameraManager = visuals.CameraManager(
                self.world, egoActor, self.hud, render_to_screen=False
            )
            self.cameraManager.set_sensor(0)

            self.semsegManager = visuals.CameraManager(self.world, egoActor, self.hud)
            self.semsegManager.set_sensor(4)

        self.world.tick()  ## allowing manualgearshift to take effect    # TODO still need this?

        for obj in self.objects:
            if isinstance(obj.carlaActor, carla.Vehicle):
                obj.carlaActor.apply_control(
                    carla.VehicleControl(manual_gear_shift=False)
                )

        self.world.tick()

        for obj in self.objects:
            if obj.speed is not None and obj.speed != 0:
                raise RuntimeError(
                    f"object {obj} cannot have a nonzero initial speed "
                    "(this is not yet possible in CARLA)"
                )

    def createObjectInSimulator(self, obj):
        if obj is None:
            warnings.warn("Requested instantiation of NoneType object")
            return

        # Extract blueprint
        try:
            blueprint = self.blueprintLib.find(obj.blueprint)
        except IndexError as e:
            found = False
            if obj.blueprint in oldBlueprintNames:
                for oldName in oldBlueprintNames[obj.blueprint]:
                    try:
                        blueprint = self.blueprintLib.find(oldName)
                        found = True
                        warnings.warn(
                            f"CARLA blueprint {obj.blueprint} not found; "
                            f"using older version {oldName}"
                        )
                        obj.blueprint = oldName
                        break
                    except IndexError:
                        continue
            if not found:
                raise SimulationCreationError(
                    f"Unable to find blueprint {obj.blueprint}" f" for object {obj}"
                ) from e
        if obj.rolename is not None:
            blueprint.set_attribute("role_name", obj.rolename)

        # set walker as not invincible
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "False")

        # Set up transform
        loc = utils.scenicToCarlaLocation(
            obj.position, world=self.world, blueprint=obj.blueprint
        )
        rot = utils.scenicToCarlaRotation(obj.orientation)
        transform = carla.Transform(loc, rot)

        # Color, cannot be set for Pedestrians
        if blueprint.has_attribute("color") and obj.color is not None:
            c = obj.color
            c_str = f"{int(c.r*255)},{int(c.g*255)},{int(c.b*255)}"
            blueprint.set_attribute("color", c_str)

        # Create Carla actor
        carlaActor = self.world.try_spawn_actor(blueprint, transform)
        if carlaActor is None:
            raise SimulationCreationError(f"Unable to spawn object {obj}")
        obj.carlaActor = carlaActor

        carlaActor.set_simulate_physics(obj.physics)

        if isinstance(carlaActor, carla.Vehicle):
            # TODO should get dimensions at compile time, not simulation time
            extent = carlaActor.bounding_box.extent
            ex, ey, ez = extent.x, extent.y, extent.z
            # Ensure each extent is positive to work around CARLA issue #5841
            obj.width = ey * 2 if ey > 0 else obj.width
            obj.length = ex * 2 if ex > 0 else obj.length
            obj.height = ez * 2 if ez > 0 else obj.height
            carlaActor.apply_control(carla.VehicleControl(manual_gear_shift=True, gear=1))
        elif isinstance(carlaActor, carla.Walker):
            carlaActor.apply_control(carla.WalkerControl())
            # spawn walker controller
            controller_bp = self.blueprintLib.find("controller.ai.walker")
            controller = self.world.try_spawn_actor(
                controller_bp, carla.Transform(), carlaActor
            )
            if controller is None:
                raise SimulationCreationError(
                    f"Unable to spawn carla controller for object {obj}"
                )
            obj.carlaController = controller

        if hasattr(obj, "tag") and obj.tag == "leadCar":
            self.lead_car = obj

        return carlaActor

    def executeActions(self, allActions):
        super().executeActions(allActions)

        # Apply control updates which were accumulated while executing the actions
        for obj in self.agents:
            ctrl = obj._control
            if ctrl is not None:
                obj.carlaActor.apply_control(ctrl)
                obj._control = None

    def step(self):
        # Run simulation for one timestep
        self.world.tick()

        # Render simulation
        if self.render:
            brake = left_blinker = right_blinker = False
            vehicle = "<UNKNOWN>"

            if self.lead_car:
                light_state = self.lead_car.carlaActor.get_light_state()
                brake = (light_state & carla.VehicleLightState.Brake) != 0
                left_blinker = (light_state & carla.VehicleLightState.LeftBlinker) != 0
                right_blinker = (light_state & carla.VehicleLightState.RightBlinker) != 0
                vehicle = self.lead_car.blueprint
            else:
                return

            light_state = (brake, left_blinker, right_blinker)

            self.cameraManager.render(self.display, light_state, vehicle, self.weather)
            self.semsegManager.render(self.display, light_state, vehicle, self.weather)

            self.frame_num += 1
            if False and self.frame_num % 1 == 0:
                self.cameraManager._render_to_screen ^= True
                self.semsegManager._render_to_screen ^= True

            if self.display is not None:
                pygame.display.flip()
            self.data_recorder.set_light_state(light_state)
            self.data_recorder.set_vehicle(vehicle)

            while self.cameraManager.has_image() and self.semsegManager.has_image():
                cam_img = self.cameraManager.parse_image()
                sem_img = self.semsegManager.parse_image()
                self.data_recorder.ingest(cam_img, sem_img)

    def getProperties(self, obj, properties):
        # Extract Carla properties
        carlaActor = obj.carlaActor
        currTransform = carlaActor.get_transform()
        currLoc = currTransform.location
        currRot = currTransform.rotation
        currVel = carlaActor.get_velocity()
        currAngVel = carlaActor.get_angular_velocity()

        # Prepare Scenic object properties
        position = utils.carlaToScenicPosition(currLoc)
        velocity = utils.carlaToScenicPosition(currVel)
        speed = math.hypot(*velocity)
        angularSpeed = utils.carlaToScenicAngularSpeed(currAngVel)
        angularVelocity = utils.carlaToScenicAngularVel(currAngVel)
        globalOrientation = utils.carlaToScenicOrientation(currRot)
        yaw, pitch, roll = obj.parentOrientation.localAnglesFor(globalOrientation)
        elevation = utils.carlaToScenicElevation(currLoc)

        values = dict(
            position=position,
            velocity=velocity,
            speed=speed,
            angularSpeed=angularSpeed,
            angularVelocity=angularVelocity,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            elevation=elevation,
        )
        return values

    def destroy(self):
        for obj in self.objects:
            if obj.carlaActor is not None:
                if isinstance(obj.carlaActor, carla.Vehicle):
                    obj.carlaActor.set_autopilot(False, self.tm.get_port())
                if isinstance(obj.carlaActor, carla.Walker):
                    obj.carlaController.stop()
                    obj.carlaController.destroy()
                obj.carlaActor.destroy()
        if self.cameraManager:
            self.cameraManager.destroy_sensor()
        if self.semsegManager:
            self.semsegManager.destroy_sensor()

        self.client.stop_recorder()

        self.world.tick()
        super().destroy()
