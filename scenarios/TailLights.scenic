import carla

param map = localPath('../../carla/Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr')
param carla_map = 'Town05'
model scenic.simulators.carla.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = Range(12, 20)
EGO_BRAKING_THRESHOLD = EGO_SPEED / 2

LEAD_CAR_SPEED = Range(5, EGO_SPEED - 2)
LEADCAR_BRAKING_THRESHOLD = 10

BRAKE_ACTION = 1.0

PERMITTED_ADV_MODELS = [
  "vehicle.audi.tt",                    # WW
  "vehicle.carlamotors.firetruck",      # WW
  "vehicle.chevrolet.impala",           # WW
  "vehicle.dodge.charger_2020",         # WW
  "vehicle.dodge.charger_police_2020",  # WW
  "vehicle.ford.ambulance",             # WW
  "vehicle.ford.crown",                 # WW
  "vehicle.lincoln.mkz_2017",           # WW
  "vehicle.lincoln.mkz_2020",           # WW
  "vehicle.mercedes.coupe_2020",        # WW
  "vehicle.mercedes.sprinter",          # WW
  "vehicle.mitsubishi.fusorosa",        # WW
  "vehicle.nissan.patrol_2021",         # WW
  "vehicle.tesla.cybertruck",           # WW
  "vehicle.tesla.model3",               # WW
  "vehicle.volkswagen.t2_2021",         # WW
#   "vehicle.audi.etron",                 # WX (lights too tiny)
#   "vehicle.dodge.charger_police",       # WX
#   "vehicle.ford.mustang",               # X (tiny blinkers)
#   "vehicle.harley-davidson.low_rider",  # X no lights
#   "vehicle.kawasaki.ninja",             # X no lights
#   "vehicle.mini.cooper_s_2021",         # X (blinkers in wrong position)
#   "vehicle.yamaha.yzf"                  # X (blinkers are tiny)
  ]

PERMITTED_LIGHT_STATES = [
    # carla.VehicleLightState.NONE,
    # carla.VehicleLightState.LeftBlinker,
    # carla.VehicleLightState.RightBlinker,
    # carla.VehicleLightState.Brake,
    # carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker,
    # carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.Brake,
    # carla.VehicleLightState.RightBlinker | carla.VehicleLightState.Brake,
    carla.VehicleLightState.LeftBlinker | carla.VehicleLightState.RightBlinker | carla.VehicleLightState.Brake,
]

## DEFINING BEHAVIORS
# EGO BEHAVIOR: Follow lane, and brake after passing a threshold distance to the leading car
behavior EgoBehavior(speed):
    try:
        do FollowLaneBehavior(speed)

    interrupt when withinDistanceToAnyCars(self, EGO_BRAKING_THRESHOLD):
        take SetBrakeAction(BRAKE_ACTION)

# LEAD CAR BEHAVIOR: Follow lane, and brake after passing a threshold distance to obstacle
behavior LeadingCarBehavior(speed, light_state):
    try: 
        take SetVehicleLightStateAction(carla.VehicleLightState(light_state))
        do FollowLaneBehavior(speed)

    interrupt when withinDistanceToAnyObjs(self, LEADCAR_BRAKING_THRESHOLD):
        take SetBrakeAction(BRAKE_ACTION)

## DEFINING SPATIAL RELATIONS

lane = Uniform(*network.lanes)

leadSpawnPoint = new OrientedPoint in lane.centerline

leadCarBlueprint = Uniform(*PERMITTED_ADV_MODELS)
leadCarLightState = Uniform(*PERMITTED_LIGHT_STATES)

leadCar = new Car at leadSpawnPoint,
        with tag "leadCar",
        with blueprint leadCarBlueprint,
        with behavior LeadingCarBehavior(LEAD_CAR_SPEED, leadCarLightState)

ego = new Car following roadDirection from leadCar for Range(-15, -10),
        with blueprint EGO_MODEL,
        with behavior EgoBehavior(EGO_SPEED)

require (distance to intersection) > 80
require (distance from leadCar to intersection) > 80
terminate when ((distance to leadCar) > 30)