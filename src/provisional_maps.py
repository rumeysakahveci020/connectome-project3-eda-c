"""PROVISIONAL mapping of super_class values that config.SUPER_CLASS_VALUES does not contain.

Not agreed by the group. It exists so the build can run; the final vocabulary is decided after the
EDA ("add MaleCNS-only levels"). Every line below is a guess from the value's name, not from the
male CNS documentation: check it and report it. Counts are what the male CNS printed.

Rules used: a value that is clearly a FlyWire level (or a '_tbc' = to-be-confirmed copy of one) is
mapped to it; nerve-cord classes and anything unclear go to 'unknown' instead of being forced into
a brain class. Nerve-cord neurons are dropped from the brain analyses anyway (in_brain).
"""

PROVISIONAL_SUPER_CLASS = {
    # shared with FlyWire v3.2.0 (FAFB has 612 sensory_ascending too): mapped, not agreed
    "sensory_ascending": "ascending",
    # central brain / optic lobe classes
    "cb_intrinsic": "central", "ol_intrinsic": "optic", "cb_motor": "motor", "cb_endocrine": "endocrine",
    "cb_sensory": "sensory", "ol_sensory": "sensory",
    "sensory_descending": "descending", "efferent_descending": "descending",
    # '_tbc' (to be confirmed) copies map to their stem
    "cb_sensory_tbc": "sensory", "visual_projection_tbc": "visual_projection", "sensory_ascending_tbc": "ascending",
    # nerve cord and unclear classes: not forced into a brain class
    "vnc_intrinsic": "unknown",       # no FAFB counterpart
    "vnc_sensory": "unknown",         # 6,365
    "vnc_motor": "unknown",           # 708
    "vnc_efferent": "unknown",        # 94
    "vnc_endocrine": "unknown",       # 22
    "vnc_sensory_tbc": "unknown",     # 9
    "vnc_tbc": "unknown",             # 1
    "ens": "unknown",                 # 47, meaning not checked
    "efferent_ascending": "unknown",  # 8, no FAFB counterpart
    "cb_efferent": "unknown",         # 4, could be motor or endocrine
}
