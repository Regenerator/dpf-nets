import re
import numpy as np

from collections import OrderedDict


class ObjMesh(object):
    def __init__(self, filename):
        self.obj_filename = filename

        # Read obj file #
        with open(self.obj_filename, 'r') as objf:
            raw_obj_data = objf.read().split('\n')
        raw_obj_data = list(map(lambda l: re.sub(' +', ' ', l.strip()), raw_obj_data))
        for line_ind in np.array(list(map(lambda l: l == '' or l[0] == '#', raw_obj_data))).nonzero()[0][::-1]:
            del raw_obj_data[line_ind]

        # Read vertices data #
        self.vertices = np.empty((0, 3), dtype=np.float32)
        self.groups = {}

        for line in raw_obj_data:
            line_segs = line.split()

            if line_segs[0] == 'v':
                self.vertices = np.append(
                    self.vertices,
                    np.array(list(map(lambda x: np.float32(x), line_segs[1:]))).reshape(1, -1),
                    axis=0
                )

            if line_segs[0] == 'g' or line_segs[0] == 'o':
                i = 0
                cur_group = '{}_{}'.format(line_segs[-1], i)
                while cur_group in self.groups:
                    i += 1
                    cur_group = '{}_{}'.format(line_segs[-1], i)
                self.groups[cur_group] = {
                    'faces_v': np.empty((0, 3), dtype=np.int32),
                    'lines': np.empty((0, 2), dtype=np.int32),
                }

            if line_segs[0] == 'f':
                if cur_group is None:
                    i = 0
                    cur_group = 'initg_{}'.format(i)
                    while cur_group in self.groups:
                        i += 1
                        cur_group = 'initg_{}'.format(i)
                    self.groups[cur_group] = {
                        'faces_v': np.empty((0, 3), dtype=np.int32),
                        'lines': np.empty((0, 2), dtype=np.int32),
                    }

                tmp = list(map(lambda l: l.split('/'), line_segs[1:]))

                self.groups[cur_group]['faces_v'] = np.append(
                    self.groups[cur_group]['faces_v'],
                    np.array(list(map(lambda l: np.int32(l[0]), tmp))).reshape(1, -1),
                    axis=0
                )

            if line_segs[0] == 'l':
                if cur_group is None:
                    i = 0
                    cur_group = 'initg_{}'.format(i)
                    while cur_group in self.groups:
                        i += 1
                        cur_group = 'initg_{}'.format(i)
                    self.groups[cur_group] = {
                        'faces_v': np.empty((0, 3), dtype=np.int32),
                        'lines': np.empty((0, 2), dtype=np.int32),
                    }

                self.groups[cur_group]['lines'] = np.append(
                    self.groups[cur_group]['lines'],
                    np.array([line_segs[1:]], dtype=np.int32).reshape(1, -1),
                    axis=0
                )

        # Shift vertex indices #
        for name, geometry in self.groups.items():
            if len(geometry['faces_v']) > 0:
                geometry['faces_v'] -= 1
            if len(geometry['lines']) > 0:
                geometry['lines'] -= 1

    def cleanup(self):
        # Remove groups without polygons #
        empty_groups = []
        for key, group in self.groups.items():
            if len(group['faces_v']) == 0:
                empty_groups.append(key)
        for empty_group_key in empty_groups:
            del self.groups[empty_group_key]

        # Remove isolated vertices #
        unique_v_inds = set()
        for group in self.groups.values():
            unique_v_inds |= set(np.unique(group['faces_v']))
        isolated_v_inds = set(np.arange(len(self.vertices))) - unique_v_inds

        if len(isolated_v_inds) > 0:
            old2new = {}
            shift = 0
            for i in range(len(self.vertices)):
                if i in isolated_v_inds:
                    shift += 1
                    old2new[i] = None
                else:
                    old2new[i] = i - shift
            old2new_func = np.vectorize(lambda ind: old2new[ind])

            self.vertices = self.vertices[sorted(list(unique_v_inds))]
            for key, group in self.groups.items():
                self.groups[key]['faces_v'] = old2new_func(group['faces_v'])

        # Remove duplicate vertices #
        duplicate2single = {}
        for i in range(len(self.vertices) - 1):
            tmp = np.sqrt(((self.vertices[(i + 1):] - self.vertices[i].reshape(1, -1))**2).sum(1))
            tmp = np.isclose(tmp, 0., atol=5e-6).nonzero()[0] + i + 1
            tmp = list(map(lambda ind: duplicate2single.update({ind: i}), tmp))

        if len(duplicate2single) > 0:
            old2new = {}
            shift = 0
            for i in range(len(self.vertices)):
                if i in duplicate2single:
                    old2new[i] = old2new[duplicate2single[i]]
                    shift += 1
                else:
                    old2new[i] = i - shift
            old2new_func = np.vectorize(lambda ind: old2new[ind])

            mask = np.ones(len(self.vertices), dtype=np.bool)
            mask[list(duplicate2single.keys())] = False
            self.vertices = self.vertices[mask]
            for key, group in self.groups.items():
                self.groups[key]['faces_v'] = old2new_func(group['faces_v'])

        # Remove non-triangular polygons #
        for key, group in self.groups.items():
            mask = list(map(lambda x: len(set(x)) == 3, group['faces_v']))
            self.groups[key]['faces_v'] = group['faces_v'][mask]

        # Remove true zero-area faces #
        for key, group in self.groups.items():
            group_ps = self.vertices[group['faces_v']]
            group_ps_areas = np.sqrt((np.cross(
                group_ps[:, 2] - group_ps[:, 0],
                group_ps[:, 1] - group_ps[:, 0]
            )**2).sum(1)) / 2.
            mask = np.logical_not(np.isclose(group_ps_areas, 0., atol=1e-10))
            self.groups[key]['faces_v'] = group['faces_v'][mask]

        # Remove faces with points on a single line #
        for key, group in self.groups.items():
            group_ps = self.vertices[group['faces_v']]
            tmp1 = group_ps[:, 1] - group_ps[:, 0]
            tmp1 /= np.sqrt((tmp1**2).sum(1)).reshape(-1, 1)
            tmp2 = group_ps[:, 2] - group_ps[:, 0]
            tmp2 /= np.sqrt((tmp2**2).sum(1)).reshape(-1, 1)
            mask = np.logical_not(np.isclose(np.fabs((tmp2 * tmp1).sum(1)), 1., rtol=1e-5))
            self.groups[key]['faces_v'] = group['faces_v'][mask]

        # Remove duplicate faces in each group #
        for key, group in self.groups.items():
            unique_faces_inds = list(OrderedDict(list(map(
                lambda i, x: (frozenset(x), i),
                range(len(group['faces_v'])), group['faces_v']
            ))).values())
            self.groups[key]['faces_v'] = group['faces_v'][unique_faces_inds]

        # Remove duplicate faces across all groups #
        # Might be color ambigous for zero-thickness parts of the shape: keeping the last read face #
        all_faces_v = np.empty((0, 3), dtype=np.int32)
        group_borders = [0]
        for group in self.groups.values():
            all_faces_v = np.append(all_faces_v, group['faces_v'], axis=0)
            group_borders.append(group_borders[-1] + len(group['faces_v']))
        group_borders = np.array(group_borders, dtype=np.int32)

        unique_faces_inds = set(OrderedDict(list(map(
            lambda i, x: (frozenset(x), i),
            range(len(all_faces_v)), all_faces_v
        ))).values())
        duplicate_faces_inds = set(np.arange(len(all_faces_v))) - unique_faces_inds

        old2new = [[] for group in self.groups.values()]
        for i in range(len(all_faces_v)):
            group_ind = (i < group_borders[1:]).argmax()
            if i not in duplicate_faces_inds:
                old2new[group_ind].append(i - group_borders[group_ind])

        for i, key_group in enumerate(self.groups.items()):
            self.groups[key_group[0]]['faces_v'] = key_group[1]['faces_v'][old2new[i]]

        # Recheck empty groups #
        empty_groups = []
        for key, group in self.groups.items():
            if len(group['faces_v']) == 0:
                empty_groups.append(key)
        for empty_group_key in empty_groups:
            del self.groups[empty_group_key]

        # Recheck isolated vertices #
        unique_v_inds = set()
        for group in self.groups.values():
            unique_v_inds |= set(np.unique(group['faces_v']))
        isolated_v_inds = set(np.arange(len(self.vertices))) - unique_v_inds

        if len(isolated_v_inds) > 0:
            old2new = {}
            shift = 0
            for i in range(len(self.vertices)):
                if i in isolated_v_inds:
                    shift += 1
                    old2new[i] = None
                else:
                    old2new[i] = i - shift
            old2new_func = np.vectorize(lambda ind: old2new[ind])

            self.vertices = self.vertices[sorted(list(unique_v_inds))]
            for key, group in self.groups.items():
                self.groups[key]['faces_v'] = old2new_func(group['faces_v'])

        # Recenter mesh to zero mean and rescale to fit into a unit sphere #
        all_faces_v = np.empty((0, 3), dtype=np.int32)
        for group in self.groups.values():
            all_faces_v = np.append(
                all_faces_v,
                group['faces_v'],
                axis=0
            )

        polygons = self.vertices[all_faces_v]
        polygons_centers = polygons.mean(1)
        areas = np.sqrt((np.cross(polygons[:, 2] - polygons[:, 0], polygons[:, 2] - polygons[:, 1])**2).sum(1)) / 2.0
        weights = areas / areas.sum()
        shape_center = (weights.reshape(-1, 1) * polygons_centers).sum(0)
        self.vertices -= shape_center.reshape(1, -1)
        shape_scale = np.sqrt((self.vertices**2).sum(1)).max()
        self.vertices /= shape_scale
        self.vertices_scale = shape_scale
        self.vertices_center = shape_center

    def reformat(self):
        faces_v = np.empty((0, 3), dtype=np.uint32)
        for group in self.groups.values():
            faces_v = np.append(faces_v, group['faces_v'], axis=0)

        mins, maxs = np.min(self.vertices, 0), np.max(self.vertices, 0)
        bbox_c = (maxs + mins) / 2.
        bbox_s = (maxs - mins).max()

        return {
            'vertices_c': self.vertices,
            'orig_c': self.vertices_center,
            'orig_s': self.vertices_scale,
            'bbox_c': bbox_c,
            'bbox_s': bbox_s,
            'faces_vc': faces_v
        }
