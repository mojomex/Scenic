import carla
from itertools import chain, combinations
from functools import reduce
from operator import or_

param map = localPath('../../carla/Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive/Town05.xodr')
param carla_map = 'Town05'

model scenic.simulators.carla.model

## CONSTANTS
EGO_MODEL = "vehicle.lincoln.mkz_2017"
EGO_SPEED = Range(12, 20)
EGO_BRAKING_THRESHOLD = EGO_SPEED / 2

ADV_SPEED = Range(5, EGO_SPEED - 2)
ADV_BRAKING_THRESHOLD = 10

BRAKE_ACTION = 1.0

PERMITTED_ADV_MODELS = [
#   "vehicle.audi.tt",                    # WX (rear lights unknown)
  "vehicle.carlamotors.firetruck",      # WW  (side spotlights count as low beams)
  "vehicle.chevrolet.impala",           # WW
  "vehicle.dodge.charger_2020",         # WW
  "vehicle.dodge.charger_police_2020",  # WW
#   "vehicle.ford.ambulance",             # WW (some lights occluded)
  "vehicle.ford.crown",                 # WW
  # "vehicle.lincoln.mkz_2017",           # WX (no front blinkers)
  "vehicle.lincoln.mkz_2020",           # WW
  "vehicle.mercedes.coupe_2020",        # WW
#   "vehicle.mercedes.sprinter",          # WW (some lights occluded)
#   "vehicle.mitsubishi.fusorosa",        # WW (some lights occluded)
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

PERMITTED_LIGHTS = [
    carla.VehicleLightState.LeftBlinker,
    carla.VehicleLightState.RightBlinker,
    carla.VehicleLightState.Brake,
    carla.VehicleLightState.LowBeam,
    carla.VehicleLightState.Reverse,
    carla.VehicleLightState.Position,
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
behavior EgoBehavior(speed):
    try:
        do FollowLaneBehavior(speed)

    interrupt when withinDistanceToAnyCars(self, EGO_BRAKING_THRESHOLD):
        take SetBrakeAction(BRAKE_ACTION)

# LEAD CAR BEHAVIOR: Follow lane, and brake after passing a threshold distance to obstacle
behavior LeadingCarBehavior(speed, light_state):
    take SetVehicleLightStateAction(carla.VehicleLightState(light_state))

## DEFINING SPATIAL RELATIONS

lane = Uniform(*network.lanes)

egoSpawnPoint = new OrientedPoint in lane.centerline
leadSpawnPoint = new Point following roadDirection from egoSpawnPoint for Range(20, 50)

advBlueprint = Uniform(*PERMITTED_ADV_MODELS)
advLightState = Uniform(*PERMITTED_LIGHTS)

adv = new Car at leadSpawnPoint, facing toward egoSpawnPoint,
        with tag "leadCar",
        with blueprint advBlueprint,
        with behavior LeadingCarBehavior(ADV_SPEED, advLightState)

ego = new Car at egoSpawnPoint,
        with blueprint EGO_MODEL,
        with behavior EgoBehavior(EGO_SPEED)

require (distance to intersection) > 80
require (distance from adv to intersection) > 80
terminate when ((distance to egoSpawnPoint) > 5 and (ego.speed < 0.1))
