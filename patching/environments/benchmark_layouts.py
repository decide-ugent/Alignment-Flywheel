# benchmark_layouts.py
# 8x8 grids -> 6x6 interior.
# 1 = Wall, 0 = Free Space, "r" = Start Region (Uniformly sampled), "g" = Goal

LAYOUTS = {
    "layout-01": {
        "name": "base_open",
        "role": "Base layout with open corridors.",
        "map": [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1,"g", 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 0, 0, 1],
            [1,"r","r", 0, 0, 1, 0, 1],
            [1,"r","r", 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "routes": {
            "route-01": {"name": "main_path", "waypoints_tiles": [(6,1), (5,1), (5,3), (3,3), (3,4), (1,4), (1,6), (2,6)]}
        }
    },
    "layout-02": {
        "name": "mid_wall_block",
        "role": "Wall added at row 3 col 3 blocks direct path through center.",
        "map": [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1,"g", 1],
            [1, 1, 0, 1, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 0, 0, 1],
            [1,"r","r", 0, 0, 1, 0, 1],
            [1,"r","r", 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "routes": {
            "route-01": {"name": "detour_path", "waypoints_tiles": [(6,1), (5,1), (5,4), (6,4), (6,6), (4,6), (4,5), (3,5), (3,4), (1,4), (1,6), (2,6)]}
        }
    },
    "layout-03": {
        "name": "upper_wall_block",
        "role": "Wall added at row 3 col 5 blocks upper passage.",
        "map": [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1,"g", 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 0, 0, 1],
            [1,"r","r", 0, 0, 1, 0, 1],
            [1,"r","r", 1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "routes": {
            "route-01": {"name": "main_path", "waypoints_tiles": [(6,1), (5,1), (5,3), (3,3), (3,4), (1,4), (1,6), (2,6)]}
        }
    },
    "layout-04": {
        "name": "lower_wall_swap",
        "role": "Wall/open swap at rows 5-6 col 3 forces bottom detour.",
        "map": [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1,"g", 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 0, 0, 1],
            [1,"r","r", 1, 0, 1, 0, 1],
            [1,"r","r", 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "routes": {
            "route-01": {"name": "bottom_detour", "waypoints_tiles": [(6,1), (6,6), (4,6), (4,5), (3,5), (3,4), (1,4), (1,6), (2,6)]}
        }
    },
    "layout-05": {
        "name": "combined_variant",
        "role": "Combined changes: upper wall + opened row 4 col 2 + lower swap.",
        "map": [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 1,"g", 1],
            [1, 1, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [1,"r","r", 1, 0, 1, 0, 1],
            [1,"r","r", 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "routes": {
            "route-01": {"name": "left_climb", "waypoints_tiles": [(6,1), (5,1), (5,2), (3,2), (3,4), (1,4), (1,6), (2,6)]}
        }
    }
}
