import carla

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

param SAFETY_DIST = 5
CRASH_DIST = 5
TERM_DIST = 70

PERMITTED_ADV_MODELS = [
    "vehicle.carlamotors.firetruck",
    "vehicle.chevrolet.impala",
    "vehicle.dodge.charger_2020",
    "vehicle.dodge.charger_police_2020",
    "vehicle.ford.crown",
    "vehicle.lincoln.mkz_2017",
    "vehicle.lincoln.mkz_2020",
    "vehicle.mercedes.coupe_2020",
    "vehicle.nissan.patrol_2021",
    "vehicle.tesla.cybertruck",
    "vehicle.tesla.model3",
    "vehicle.volkswagen.t2_2021",
]

# Vehicles either do not indicate or indicate correctly.
# Independently of the indicator state, vehicles may brake.
PERMITTED_LIGHT_STATES = {
    ManeuverType.LEFT_TURN: [
        carla.VehicleLightState.NONE,
        carla.VehicleLightState.LeftBlinker,
        carla.VehicleLightState.Brake,
        carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.Brake,
    ], 
    ManeuverType.RIGHT_TURN: [
        carla.VehicleLightState.NONE,
        carla.VehicleLightState.RightBlinker,
        carla.VehicleLightState.Brake,
        carla.VehicleLightState.RightBlinker | carla.VehicleLightState.Brake,
    ]
}

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

advTurnDirection = Uniform(ManeuverType.LEFT_TURN, ManeuverType.RIGHT_TURN).sample()

advBlueprint = Uniform(*PERMITTED_ADV_MODELS)
advLightState = Uniform(*PERMITTED_LIGHT_STATES[advTurnDirection])

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
advManeuver = Uniform(*filter(lambda m: m.type is advTurnDirection, advInitLane.maneuvers))
advTrajectory = [advInitLane, advManeuver.connectingLane, advManeuver.endLane]
advSpawnPt = new OrientedPoint in advInitLane.centerline

#################################
# SCENARIO SPECIFICATION        #
#################################

ego = new Car at egoSpawnPt,
    with blueprint EGO_MODEL,
    with behavior EgoBehavior(globalParameters.EGO_SPEED, egoTrajectory)

adversary = new Car at advSpawnPt,
    with tag "leadCar",
    with blueprint advBlueprint,
    with behavior LeadingCarBehavior(globalParameters.ADV_SPEED, advTrajectory, advLightState)

require EGO_INIT_DIST[0] <= (distance to intersection) <= EGO_INIT_DIST[1]
require ADV_INIT_DIST[0] <= (distance from adversary to intersection) <= ADV_INIT_DIST[1]
require advLightState in PERMITTED_LIGHT_STATES[advTurnDirection]
terminate when (distance to egoSpawnPt) > TERM_DIST