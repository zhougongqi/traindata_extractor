import os

home_dir = os.path.expanduser("~")

# for trainData
MYS_state_list = [
    "Johor",
    "Kedah",
    "Kelantan",
    "Melakas",
    "Negeri Sembilan",
    "Pahang",
    "Perak",
    "Perlis",
    "Pulau Pinang",
    "Sabah",
    "Sarawak",
    "Selangor",
    "Trengganu",
    "peninsula",
    "kalimantan",
    "Overall",
]
MYS_state_abbr = [
    "JH",
    "KD",
    "KL",
    "ML",
    "NS",
    "PH",
    "PR",
    "PL",
    "PP",
    "SB",
    "SR",
    "SL",
    "TR",
    "PE",
    "KA",
    "OV",
]
IDN_state_list = [
    "Aceh",
    "Bangka-Belitung",
    "Benkulu",
    "Jambi",
    "Kalimantan Barat",
    "Kalimantan Selatan",
    "Kalimantan Tengah",
    "Kalimantan Timur",
    "Kalimantan Utara",
    "Kepulauan Riau",
    "Lampung",
    "Riau",
    "Sumatera Barat",
    "Sumatera Selatan",
    "Sumatera Utara",
    "Sumatera",
    "Kalimantan",
    "Overall",
]
IDN_state_abbr = [
    "AC",
    "BB",
    "BK",
    "JB",
    "KB",
    "KS",
    "KT",
    "KI",
    "KU",
    "KR",
    "LP",
    "RI",
    "SB",
    "SS",
    "SU",
    "SM",
    "KA",
    "OV",
]
MYS_label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1, -2, -3]
MYS_abbr = {abbr: name for abbr, name in zip(MYS_state_abbr, MYS_state_list)}
MYS_id_abbr = {abbr: label for abbr, label in zip(MYS_state_abbr, MYS_label)}
MYS_id_name = {name: label for name, label in zip(MYS_state_list, MYS_label)}
MYS_id = {**MYS_id_abbr, **MYS_id_name}

IDN_label = [1, 3, 5, 9, 13, 14, 15, 16, 17, 18, 19, 25, 31, 32, 33, -1, -2, -3]
IDN_abbr = {abbr: name for abbr, name in zip(IDN_state_abbr, IDN_state_list)}
IDN_id_abbr = {abbr: label for abbr, label in zip(IDN_state_abbr, IDN_label)}
IDN_id_name = {name: label for name, label in zip(IDN_state_list, IDN_label)}
IDN_id = {**IDN_id_abbr, **IDN_id_name}

MYS_id_group = {
    -1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13],
    -2: [10, 11],
    -3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
}
IDN_id_group = {
    -1: [1, 3, 5, 9, 18, 19, 25, 31, 32, 33],
    -2: [13, 14, 15, 16, 17],
    -3: [1, 3, 5, 9, 13, 14, 15, 16, 17, 18, 19, 25, 31, 32, 33],
}

STATE_id = {
    "MYS": MYS_id,
    "IDN": IDN_id,
    "MYS_group": MYS_id_group,
    "IDN_group": IDN_id_group,
}

label_Dict = {1: "palm", 2: "farm", 3: "otree", 4: "city", 5: "other"}

MPOB_2017 = [
    5811145,
    2708413,
    748860,
    87538,
    158310,
    57372,
    184815,
    741495,
    406469,
    660,
    13563,
    1546904,
    1555828,
    137783,
    171548,
]
MPOB_2016 = [
    5737985,
    2679502,
    745630,
    87786,
    155458,
    56149,
    178958,
    732052,
    397908,
    652,
    14135,
    1551714,
    1506769,
    138831,
    171943,
]

# run for model training and product
poly_unique_id = "id"  # "OBJECTID_1"
field_name_state = "ID_1"

# run for model data preprocess
aux_data = {
    "home_dir": home_dir,
    "prj_path": home_dir + "/data_pool/Ray_EX/PRJ_FILE/palm_wgs84.prj",
    "band_list_dict": {
        "MOD09Q1": {},
        "MYD09Q1": {},
        "MOD13Q1": {
            "EVI": [-3000],
            "NDVI": [-3000],
            "blue": [-1000],
            "red": [-1000],
            "NIR": [-1000],
            "MIR": [-1000],
        },
        "SRTM1": {"DEM": [-32768]},
        "sentinel1": {"VV": [10000], "VH": [10000]},
    },
    "STATE_id": STATE_id,
    "poly_unique_id": poly_unique_id,
    "field_name_state": field_name_state,
    "label_dict": label_Dict,
}


# # state name to be processed in traindata collection
# bm_list = [["subtraction", "VV", "VH"], ["division", "VV", "VH"]]  # band math list
# bm_strlist = ["VV-VH", "VV/VH"]
# morph_list = [["Open", 5], ["Close", 5], ["Erode", 2], ["Dilate", 2]]
# # morphology string, all seperators in ['_','-','.'] or their combination can be used
# # freely as you wish to make the expression easy to read, since all the seperators
# # will be removed in parse function.
# morph_str = "O5-C5-E_2-D.2"
