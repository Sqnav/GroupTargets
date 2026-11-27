"""Deterministic non-colocated sensor transforms, never learned translations.

Station axes are X forward, Y right, Z up. Its array_basis is an orthogonal
coordinate basis, possibly det=-1, and must never be converted as an SO(3)
quaternion. Positions/vectors are row vectors at this public NumPy interface.
"""
from dataclasses import dataclass, replace
from typing import Optional
import numpy as np
from .types import Observations


def _points(value,name='Positions'):
    points=np.asarray(value,dtype=float)
    if points.ndim<1 or points.shape[-1]!=3 or not np.isfinite(points).all():
        raise ValueError(f'{name} must be finite (...,3) vectors')
    return points


def _basis(value,name='Coordinate basis'):
    matrix=np.asarray(value,dtype=float)
    if (matrix.shape!=(3,3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix.T@matrix,np.eye(3),atol=1e-7,rtol=0)):
        raise ValueError(f'{name} must be an orthogonal 3x3 matrix')
    return matrix


@dataclass(frozen=True)
class FixedStationFrame:
    origin_ned: np.ndarray
    basis_to_ned: np.ndarray
    episode_id: Optional[str] = None

    def __post_init__(self):
        origin=_points(self.origin_ned,'Station origin')
        if origin.shape!=(3,):raise ValueError('Station origin must be a 3-vector')
        basis=_basis(self.basis_to_ned,'Station array basis')
        # Keep the reference immutable even if callers mutate input arrays.
        origin=origin.copy();basis=basis.copy()
        origin.setflags(write=False);basis.setflags(write=False)
        object.__setattr__(self,'origin_ned',origin)
        object.__setattr__(self,'basis_to_ned',basis)

    @classmethod
    def from_observation(cls,observation):
        if not hasattr(observation,'station_position') or not hasattr(observation,'array_basis'):
            raise ValueError('Calibrated station_position and array_basis are required')
        return cls(observation.station_position,observation.array_basis,getattr(observation,'episode_id',None))

    def validate_observation(self,observation):
        """This adapter supports a single fixed site/basis throughout a group."""
        if not isinstance(observation,Observations):raise TypeError('Expected Observations')
        if self.episode_id is not None and observation.episode_id!=self.episode_id:
            raise ValueError('A new episode requires a new fixed station frame')
        station=_points(observation.station_position,'Station position')
        basis=_basis(observation.array_basis,'Station array basis')
        if (station.shape!=(3,) or not np.allclose(station,self.origin_ned,atol=1e-6,rtol=0)
                or not np.allclose(basis,self.basis_to_ned,atol=1e-8,rtol=0)):
            raise ValueError('Station position and array basis must remain fixed within an episode')

    def to_local_positions(self,positions):
        return (_points(positions)-self.origin_ned)@self.basis_to_ned

    def to_world_positions(self,positions):
        return _points(positions)@self.basis_to_ned.T+self.origin_ned

    def to_local_vectors(self,vectors):
        return _points(vectors,'Vectors')@self.basis_to_ned

    def to_world_vectors(self,vectors):
        return _points(vectors,'Vectors')@self.basis_to_ned.T

    def to_world_covariance(self,covariance):
        covariance=np.asarray(covariance,float)
        if (covariance.ndim<2 or covariance.shape[-2:] not in ((3,3),(6,6))
                or not np.isfinite(covariance).all()
                or not np.allclose(covariance,np.swapaxes(covariance,-1,-2),atol=1e-7,rtol=1e-7)):
            raise ValueError('Expected finite symmetric (...,3,3) or (...,6,6) covariance')
        transform=self.basis_to_ned
        if covariance.shape[-1]==6:
            transform=np.zeros((6,6));transform[:3,:3]=self.basis_to_ned;transform[3:,3:]=self.basis_to_ned
        return transform@covariance@transform.T

    def observations(self,observation):
        """Return sensor-only observation in this stationary array frame.

        Camera and optional observer body poses are transformed independently;
        their FRD bases can become det=-1. RGB/IQ/calibration samples are shared
        read-only by convention, and no Targets or target labels are consumed.
        """
        self.validate_observation(observation)
        camera=replace(observation.camera,
            position=self.to_local_positions(observation.camera.position),
            rotation=self.basis_to_ned.T@_basis(observation.camera.rotation,'Camera basis'))
        observer=observation.observer
        if observer is not None:
            observer=replace(observer,position=self.to_local_positions(observer.position),
                rotation=self.basis_to_ned.T@_basis(observer.rotation,'Observer body basis'))
        reference=observation.observer_reference
        if reference is not None:
            transform=np.zeros((6,6));transform[:3,:3]=self.basis_to_ned.T;transform[3:,3:]=self.basis_to_ned.T
            reference=replace(reference,position=self.to_local_positions(reference.position),
                velocity=None if reference.velocity is None else self.to_local_vectors(reference.velocity),
                covariance=None if reference.covariance is None else transform@reference.covariance@transform.T,
                position_covariance=None if reference.position_covariance is None else
                    self.basis_to_ned.T@reference.position_covariance@self.basis_to_ned)
        return replace(observation,camera=camera,observer=observer,
                       observer_reference=reference,
                       station_position=np.zeros(3),array_basis=np.eye(3))


def spherical_from_local(points):
    """Array-front XYZ -> metres and degree azimuth/elevation (positive up).

    The source single-array model only covers x>0. Invalid or uninitialized
    states must be handled as such by the caller, never converted to fake zeros.
    """
    points=_points(points)
    if points.ndim!=2:raise ValueError('Expected points [N,3]')
    if np.any(points[:,0]<=0):raise ValueError('Spherical targets must lie in the array forward hemisphere')
    ranges=np.hypot(np.hypot(points[:,0],points[:,1]),points[:,2])
    if not np.isfinite(ranges).all() or np.any(ranges<=0):
        raise ValueError('Spherical targets require finite positive range')
    azimuth=np.rad2deg(np.arctan2(points[:,1],points[:,0]))
    elevation=np.rad2deg(np.arctan2(points[:,2],np.hypot(points[:,0],points[:,1])))
    return [dict(range_m=float(r),azimuth_deg=float(a),elevation_deg=float(e))
            for r,a,e in zip(ranges,azimuth,elevation)]


def local_from_spherical(records):
    """Inverse of spherical_from_local; only three public estimate keys read."""
    if not isinstance(records,(list,tuple)):raise TypeError('Expected a sequence of spherical target records')
    if not records:return np.empty((0,3))
    try:
        values=np.array([[item['range_m'],item['azimuth_deg'],item['elevation_deg']] for item in records],float)
    except (KeyError,TypeError,ValueError) as exc:
        raise ValueError('Each spherical record requires numeric range_m, azimuth_deg, elevation_deg') from exc
    if (not np.isfinite(values).all() or np.any(values[:,0]<=0)
            or np.any(abs(values[:,1:])>=90.)):
        raise ValueError('Require finite positive range and front-hemisphere azimuth/elevation in (-90,90) degrees')
    ranges=values[:,0];az,el=np.deg2rad(values[:,1]),np.deg2rad(values[:,2])
    return ranges[:,None]*np.column_stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])


def world_to_body_positions(points,bodypose):
    """World NED -> measured observer-body FRD, using its actual body origin."""
    if bodypose is None:raise ValueError('Known observer body pose required; camera pose cannot substitute')
    position=_points(bodypose.position,'Observer body position')
    if position.shape!=(3,):raise ValueError('Observer body position must be a 3-vector')
    return (_points(points)-position)@_basis(bodypose.rotation,'Observer body basis')
