import carla
from itertools import chain, combinations
from functools import reduce
from operator import or_

param map = localPath('../../carla/Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr')
param carla_map = 'Town05'

model scenic.simulators.carla.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz_2017"
BRAKE_ACTION = 1.0

EGO_INIT_DIST = [0, 7]
param EGO_SPEED = VerifaiRange(0, 5)
param EGO_BRAKE = VerifaiRange(0.5, 1.0)

ADV_INIT_DIST = [0, 7]
param ADV_SPEED = VerifaiRange(0, 10)

param SAFETY_DIST = 7
CRASH_DIST = 5
TERM_DIST = 70

PERMITTED_ADV_MODELS = [
    "vehicle.carlamotors.firetruck",
    "vehicle.chevrolet.impala",
    "vehicle.dodge.charger_2020",
    "vehicle.dodge.charger_police_2020",
    "vehicle.ford.crown",
    # "vehicle.lincoln.mkz_2017",  # no front blinkers
    "vehicle.lincoln.mkz_2020",
    "vehicle.mercedes.coupe_2020",
    "vehicle.nissan.patrol_2021",
    "vehicle.tesla.cybertruck",
    "vehicle.tesla.model3",
    "vehicle.volkswagen.t2_2021",
]

# Vehicles either do not indicate or indicate correctly.
# Independently of the indicator state, vehicles may brake.
PERMITTED_LIGHTS = [
    carla.VehicleLightState.LeftBlinker,
    carla.VehicleLightState.RightBlinker,
    carla.VehicleLightState.Brake,
    carla.VehicleLightState.LowBeam,
    carla.VehicleLightState.Reverse,
    carla.VehicleLightState.HighBeam,
    carla.VehicleLightState.Interior,
    carla.VehicleLightState.Special1,
    carla.VehicleLightState.Special2,
]

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

PERMITTED_LIGHTS = [reduce(or_, combo, carla.VehicleLightState.NONE) for combo in powerset(PERMITTED_LIGHTS)]

## DEFINING BEHAVIORS
# EGO BEHAVIOR: Follow lane, and brake after passing a threshold distance to the leading car
behavior EgoBehavior(speed, trajectory):
    try:
        do FollowTrajectoryBehavior(target_speed=speed, trajectory=trajectory)
    interrupt when withinDistanceToAnyObjs(self, globalParameters.SAFETY_DIST):
        take SetBrakeAction(globalParameters.EGO_BRAKE)
    interrupt when withinDistanceToAnyObjs(self, CRASH_DIST):
        terminate

# LEAD CAR BEHAVIOR: Follow lane, and brake after passing a threshold distance to obstacle
behavior LeadingCarBehavior(speed, trajectory, light_state):
    take SetVehicleLightStateAction(carla.VehicleLightState(light_state))
    do FollowTrajectoryBehavior(target_speed=speed, trajectory=trajectory)

## DEFINING SPATIAL RELATIONS

advBlueprint = Uniform(*PERMITTED_ADV_MODELS)
advLightState = Uniform(*PERMITTED_LIGHTS)

#################################
# SPATIAL RELATIONS             #
#################################

intersection = Uniform(*filter(lambda i: i.is4Way, network.intersections))

egoInitLane = Uniform(*intersection.incomingLanes)
egoManeuver = Uniform(*filter(lambda m: m.type is ManeuverType.STRAIGHT, egoInitLane.maneuvers))
egoTrajectory = [egoInitLane, egoManeuver.connectingLane, egoManeuver.endLane]
egoSpawnPt = new OrientedPoint in egoInitLane.centerline

advInitLane = Uniform(*filter(lambda m:
        m.type is ManeuverType.STRAIGHT,
        egoManeuver.reverseManeuvers)
    ).startLane
advManeuver = Uniform(*advInitLane.maneuvers)
advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]
advSpawnPt = new OrientedPoint in advInitLane.centerline

#################################
# SCENARIO SPECIFICATION        #
#################################

ego = new Car at egoSpawnPt,
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(globalParameters.EGO_SPEED, egoTrajectory)

adv = new Car at advSpawnPt,
    with tag "leadCar",
    with blueprint advBlueprint,
    with behavior LeadingCarBehavior(globalParameters.ADV_SPEED, advTrajectory, advLightState)

require EGO_INIT_DIST[0] <= (distance to intersection) <= EGO_INIT_DIST[1]
require ADV_INIT_DIST[0] <= (distance from adv to intersection) <= ADV_INIT_DIST[1]
terminate when (distance to egoSpawnPt) > TERM_DIST