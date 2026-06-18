Compares a scene sampling a bunny area light (path_scene_area_emitter) with one using the spherical rectangles of the visible faces of the 
aabb of the bunny (path_scene_areaprojected_emitter).

Both converge, but the spherical one with a bit less variance (a bit more time, 0.78 vs 0.68 s)

bunny from https://github.com/alecjacobson/common-3d-test-models/tree/master/data

Added another test with the "beast" model too. Rationale: I think it will be much worse, since it has a worse real to aabb solid angle ratio 
(meaning: it has long arms, so there is a lot of empty space in the aabb, where rays will miss, so area sampling may be better here)

