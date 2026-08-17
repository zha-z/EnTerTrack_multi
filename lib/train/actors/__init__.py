from .base_actor import BaseActor
#from .ostrack import OSTrackActor
#from .ostrack_distill_b import OSTrackdistillActor


def EnTeRTrackActorThreeMDOT(*args, **kwargs):
    from .entertrack_threemdot import EnTeRTrackActorThreeMDOT as _actor
    return _actor(*args, **kwargs)


def EnTeRTrackActorTeacher(*args, **kwargs):
    from .entertrack_teacher import EnTeRTrackActorTeacher as _actor
    return _actor(*args, **kwargs)
