from . import cifti2
from ..nifti1 import intent_codes

from collections import namedtuple, OrderedDict
from itertools import chain
import numpy as np

IndexToVertex = namedtuple(
    'IndexToVertex', ['index', 'structure', 'vertex']
)

IndexToVoxel = namedtuple(
    'IndexToVoxel', ['index', 'structure', 'i', 'j', 'k', 'x', 'y', 'z']
)

IndexToNamedMap = namedtuple(
    'IndexToNamedMap', ['index', 'map_name']
)

PositionVertex = namedtuple(
    'PositionVertex', ['structure', 'vertex']
)

PositionVoxel = namedtuple(
    'PositionVoxel', ['structure', 'i', 'j', 'k']
)

LabelToNameColor = namedtuple(
    'LabelToNameColor', ['label', 'name', 'red', 'green', 'blue', 'alpha']
)


def voxel_indices_ijk_to_list(voxel_indices_ijk, matrix):
    ijk = np.array(voxel_indices_ijk).T
    xyz = np.dot(matrix[:3, :3], ijk) + matrix[:3, -1][:, None]
    return (
        ijk[0], ijk[1], ijk[2],
        xyz[0], xyz[1], xyz[2]
    )


def matrix_index_map_brain_models_to_list(matrix_index_map):
    '''
    From a Matrix Index Map object that contains brain models
    it returns a list of triples:
    (index in the array, CIFTI Structure, index in the structure)
    '''
    if (
        matrix_index_map.indices_map_to_data_type !=
        'CIFTI_INDEX_TYPE_BRAIN_MODELS'
    ):
        raise ValueError('This only works for Brain Model index maps')

    if (
        (matrix_index_map.volume is not None) and
        (matrix_index_map.volume.transformation_matrix_voxel_indices_ijk_to_xyz
            is not None)
    ):
        volume = matrix_index_map.volume
        transformation = volume.transformation_matrix_voxel_indices_ijk_to_xyz
        matrix = transformation.matrix
        if transformation.meter_exponent != -3:
            matrix = matrix * 10 ** (-3 - transformation.meter_exponent)
    else:
        matrix = np.eye(4)

    bms_vertices = chain(*(
        (IndexToVertex(*x) for x in zip(
            np.arange(bm.index_count) + bm.index_offset,
            (bm.brain_structure,) * bm.index_count,
            np.array(bm.vertex_indices)
        ))
        for bm in matrix_index_map.brain_models
        if bm.model_type == "CIFTI_MODEL_TYPE_SURFACE"
    ))

    bms_voxels = chain(*(
        (IndexToVoxel(*x) for x in zip(
            np.arange(bm.index_count) + bm.index_offset,
            (bm.brain_structure,) * bm.index_count,
            *voxel_indices_ijk_to_list(bm.voxel_indices_ijk, matrix)
        ))
        for bm in matrix_index_map.brain_models
        if bm.model_type == "CIFTI_MODEL_TYPE_VOXELS"
    ))

    bms = chain(bms_vertices, bms_voxels)
    index = sorted(bms, key=lambda e: e.index)
    return index


def matrix_index_map_parcels_to_list(matrix_index_map):
    '''
    From a Matrix Index Map object that contains parcels models
    it returns a list of triples:
    (index in the array, CIFTI Structure, index in the structure)
    '''
    if (
        matrix_index_map.indices_map_to_data_type !=
        'CIFTI_INDEX_TYPE_PARCELS'
    ):
        raise ValueError('This only works forn Brain Model index maps')
    parcels = []
    for i, parcel in enumerate(matrix_index_map.parcels):
        positions = []
        if parcel.vertices is not None:
            for v in parcel.vertices:
                positions += [(v.brain_structure, v_) for v_ in v]
        elif parcel.voxel_indices_ijk is not None:
            for v in parcel.voxel_indices_ijk:
                positions = list(v)
        else:
            raise ValueError('Parcel without spatial description')
        parcels.append(
            (i, parcel.name, positions)
        )
    return parcels


class MatrixIndexMap:
    def __init__(self, matrix_index_map):
        self.is_brain_models = False
        self.is_parcels = False
        self.is_series = False
        self.is_scalars = False
        self.is_labels = False

        if matrix_index_map.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_BRAIN_MODELS':
            self.is_brain_models = True
            self.index = matrix_index_map_brain_models_to_list(matrix_index_map)
            self._value_map = {
                (v.structure, v.vertex): v
                for v in self.index
                if isinstance(v, IndexToVertex)
            }
            self._value_map.update({
                (v.structure, v.i, v.j, v.k): v
                for v in self.index
                if isinstance(v, IndexToVoxel)
            })
        elif matrix_index_map.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_PARCELS':
            self.is_parcels = True
            self.index = matrix_index_map_parcels_to_list(matrix_index_map)
            self._value_map = {
                (v.structure, v.vertex): v
                for v in self.index
                if isinstance(v, IndexToVertex)
            }
            self._value_map.update({
                (v.structure, v.i, v.j, v.k): v
                for v in self.index
                if isinstance(v, IndexToVoxel)
            })
        elif matrix_index_map.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_LABELS':
            self.is_labels = True
            self.index = []
            self.label_tables = {}
            for index, named_map in enumerate(matrix_index_map.named_maps):
                self.index.append(IndexToNamedMap(index, named_map.map_name))
                label_table = {}
                for v in named_map.label_table:
                    label_table[v] = LabelToNameColor(
                        v, named_map.label_table[v].label, named_map.label_table[v].red,
                        named_map.label_table[v].green, named_map.label_table[v].blue,
                        named_map.label_table[v].alpha
                    )
                self.label_tables[index] = label_table

        self._index_map = OrderedDict({
            v.index: v
            for v in self.index
        })

    @property
    def indices(self):
        return self._index_map.keys()

    @property
    def mappings(self):
        return self.get_mapping_from_index(self.indices)

    def get_mapping_from_index(self, index):
        if isinstance(index, int):
            return self._index_map[index]
        else:
            return [
                self._index_map[i]
                for i in index
            ]

    def get_label_table_from_index(self, index):
        return self.label_tables[index]

    def get_index_from_mapping(self, structure, vertex=None, i=None, j=None, k=None):
        if vertex is None and i is None and j is None and k is None:
            return [
                v[1:] for v in self.index
                if v.structure == structure
            ]
        if vertex is not None and i is None and j is None and k is None:
            return self._value_map[(structure, vertex)]
        elif i is not None and j is not None and k is not None:
            return self._value_map[(structure, i, j, k)]
        else:
            raise


def cifti_from_pandas_cortex(
    data, intent=intent_codes.code['NIFTI_INTENT_NONE'],
    number_of_vertices=32492
):
    surfaces = [
        s for s in data.index.levels[0]
        if (
            s in nib.cifti2.CIFTI_BRAIN_STRUCTURES and
            'CORTEX' in s
        )
    ]

    data_to_save = []
    offset = 0
    bms = []
    for surface in surfaces:
        data_surface = data.loc[surface]
        indices = ' '.join(str(i) for i in data_surface.index)
        vi = nib.cifti2.Cifti2VertexIndices(indices=data_surface.index)
        bm = nib.cifti2.Cifti2BrainModel(
            index_offset=offset, index_count=len(vi), model_type='CIFTI_MODEL_TYPE_SURFACE',
            brain_structure=surface, n_surface_vertices=number_of_vertices,
            vertex_indices=vi
        )
        offset += len(vi)
        data_to_save.append(data_surface.values)
        bms.append(bm)

    mim_surface = nib.cifti2.Cifti2MatrixIndicesMap([1], "CIFTI_INDEX_TYPE_BRAIN_MODELS", maps=bms)
    mim_scalars = nib.cifti2.Cifti2MatrixIndicesMap(
        [0],
        "CIFTI_INDEX_TYPE_SCALARS",
        maps=[nib.cifti2.Cifti2NamedMap(map_name=str(c)) for c in data.columns]
    )
    matrix = nib.cifti2.Cifti2Matrix()
    matrix.append(mim_surface)
    matrix.append(mim_scalars)
    header = nib.cifti2.Cifti2Header(matrix=matrix)
    new_cifti = nib.Cifti2Image(np.vstack(data_to_save).T, header=header)
    new_cifti.nifti_header.set_intent(intent)
    return new_cifti
