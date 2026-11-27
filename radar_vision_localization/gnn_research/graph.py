"""Sparse member/candidate/native-evidence graph with RF-scaled soft edges.

RA and RE remain two projections of one IQ, never independent likelihoods.
The rectangular-response kernel defines a query-conditioned main-lobe proxy,
not an exact PSF of the max-projected, learned CNN maps. Weak valid pixels are
retained; no peak detector, source identity or one-target-per-cell rule is used.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from rvloc.legacy import activate
activate()
from depth_localization.geometry import candidate_geometry

C = 299792458.0
VARIANTS = ('shared','isolated','shuffled')


def rf_scales(rf=None):
    value=dict(rf or {})
    fc=float(value.get('carrier_hz',4.9e9));bw=float(value.get('bandwidth_hz',32e6))
    shape=value.get('shape',[40,128,16,16]);wave=C/fc
    dy=float(value.get('spacing_y_m',wave/2));dz=float(value.get('spacing_z_m',wave/2))
    if len(shape)!=4 or min(fc,bw,dy,dz,*shape)<=0:raise ValueError('Explicit positive native RF geometry required')
    return dict(range_first_null_m=C/(2*bw),uy_first_null=wave/(int(shape[2])*dy),
                uz_first_null=wave/(int(shape[3])*dz),frequency_samples=int(shape[1]),
                array_y=int(shape[2]),array_z=int(shape[3]),carrier_hz=fc,bandwidth_hz=bw,
                spacing_y_m=dy,spacing_z_m=dz,
                normalization='rectangular finite-frequency/array first null; query-conditioned projection proxy',
                support_first_nulls=1.)


def _kernel(x,n):
    return (torch.sinc(x)/torch.sinc(x/n).clamp_min(torch.finfo(x.dtype).eps)).square()


def _scatter_sum(value,index,size):
    return value.new_zeros((size,)+value.shape[1:]).index_add(0,index,value)


def _per_frame_sum(value,index,b):
    return _scatter_sum(value,index,b)


def build_graph(geometry,ra_map,re_map,range_axis,azimuth_axis,elevation_axis,
                ra_valid,re_valid,member_count,scales,variant='shared'):
    """Deduplicate real CNN grid locations, not sampled candidate copies.

    ``geometry`` contains [B,N,K] coordinates from trusted candidate_geometry.
    Edges are sparse: first select native range samples inside the main lobe,
    then inspect only their angular row (no dense candidate×full-map MLP).
    Isolated control duplicates a physical pixel per member, keeping all
    candidates of one member connected. Shuffled control preserves every
    candidate out-degree and every evidence in-degree while leaving evidence
    features fixed. Its edge attributes intentionally travel with source edges.
    """
    if variant not in VARIANTS:raise ValueError('Unknown graph variant')
    b,n,k=geometry['ranges'].shape;q=b*n*k;nr=len(range_axis);na=len(azimuth_axis);ne=len(elevation_axis)
    if member_count!=n:raise ValueError('Member padding dimension differs from candidate geometry')
    area=nr*(na+ne);device=ra_map.device;dtype=ra_map.dtype
    ranges=geometry['ranges'].reshape(-1);az=geometry['azimuth'].reshape(-1);el=geometry['elevation'].reshape(-1)
    valid=geometry['valid'].reshape(-1)
    owners=torch.arange(b*n,device=device).repeat_interleave(k)
    frame=owners//n
    dr=(range_axis[None]-ranges[:,None])/scales['range_first_null_m']
    range_pairs=torch.nonzero(valid[:,None] & (dr.abs()<1.),as_tuple=False)
    edge_c=[];edge_gid=[];edge_attr=[];edge_frame=[];edge_view=[]
    for view,(angles,masks,width) in enumerate(((azimuth_axis,ra_valid,na),(elevation_axis,re_valid,ne))):
        cc,rr=range_pairs[:,0],range_pairs[:,1]
        angle=angles[None].expand(len(cc),-1)
        ac=az[cc,None];ec=el[cc,None]
        if view==0:
            # True-angle RA slice: fix candidate elevation, not u=sin(az).
            duy=ec.cos()*(angle.sin()-ac.sin())/scales['uy_first_null']
            duz=torch.zeros_like(duy)
            center=az[cc]
        else:
            # True-angle RE slice: both array direction cosines change.
            duy=(angle.cos()-ec.cos())*ac.sin()/scales['uy_first_null']
            duz=(angle.sin()-ec.sin())/scales['uz_first_null']
            center=el[cc]
        rf=frame[cc]
        in_axis=(center>=angles[0])&(center<=angles[-1])&(ranges[cc]>=range_axis[0])&(ranges[cc]<=range_axis[-1])
        allowed=in_axis[:,None] & (duy.abs()<1.) & (duz.abs()<1.) & masks[rf,rr]
        pair=torch.nonzero(allowed,as_tuple=False)
        rows,aa=pair[:,0],pair[:,1];cidx=cc[rows];ridx=rr[rows]
        dy=duy[rows,aa];dz=duz[rows,aa];delta_r=dr[cidx,ridx]
        weight=_kernel(delta_r,scales['frequency_samples'])*_kernel(dy,scales['array_y'])*_kernel(dz,scales['array_z'])
        offset=0 if view==0 else nr*na
        gid=frame[cidx]*area+offset+ridx*width+aa
        flags=torch.zeros((len(rows),2),device=device,dtype=dtype);flags[:,view]=1.
        edge_c.append(cidx);edge_gid.append(gid)
        edge_attr.append(torch.cat((delta_r[:,None],dy[:,None],dz[:,None],weight[:,None],flags),-1))
        edge_frame.append(frame[cidx]);edge_view.append(torch.full_like(cidx,view))
    cidx=torch.cat(edge_c);gid=torch.cat(edge_gid);attributes=torch.cat(edge_attr)
    ef=torch.cat(edge_frame);ev=torch.cat(edge_view)
    key=gid if variant!='isolated' else gid*(b*n+1)+owners[cidx]
    unique,eidx=torch.unique(key,sorted=True,return_inverse=True)
    node_grid=unique if variant!='isolated' else unique//(b*n+1)
    all_features=torch.cat((ra_map.permute(0,2,3,1).reshape(b,-1,ra_map.shape[1]),
                            re_map.permute(0,2,3,1).reshape(b,-1,re_map.shape[1])),1).reshape(-1,ra_map.shape[1])
    features=all_features[node_grid]
    node_frame=node_grid//area;local=node_grid%area;is_re=local>=nr*na
    ri=torch.where(is_re,(local-nr*na)//ne,local//na)
    ai=torch.where(is_re,(local-nr*na)%ne,local%na)
    angle=torch.where(is_re,elevation_axis[ai.clamp(0,ne-1)],azimuth_axis[ai.clamp(0,na-1)])
    coords=torch.stack((range_axis[ri],angle.sin(),angle.cos(),is_re.to(dtype),~is_re),-1)
    original_eidx=eidx
    if variant=='shuffled':
        eidx=eidx.clone()
        # Canonical physical ordering removes arbitrary detection row order
        # from the negative control (ties describe identical coordinates).
        for f in range(b):
            for view in (0,1):
                positions=torch.nonzero((ef==f)&(ev==view),as_tuple=False).flatten()
                order=positions
                for field in (el[cidx],az[cidx],ranges[cidx],gid):
                    order=order[torch.argsort(field[order],stable=True)]
                if len(order)>1:eidx[order]=torch.roll(original_eidx[order],len(order)//2+1)
    count=len(unique)
    degree=_scatter_sum(torch.ones_like(eidx,dtype=dtype),eidx,count)
    owner_pair=torch.unique(eidx*(b*n)+owners[cidx])
    member_degree=_scatter_sum(torch.ones_like(owner_pair,dtype=dtype),owner_pair//(b*n),count)
    candidate_degree=_scatter_sum(torch.ones_like(cidx,dtype=dtype),cidx,q)
    shared_edge=(member_degree[eidx]>=2).to(dtype)
    shared_candidate=_scatter_sum(shared_edge,cidx,q)>0
    usable=candidate_degree>0
    shared_members=(_scatter_sum(shared_candidate.to(dtype),owners,b*n)>0)
    valid_members=geometry['valid'].any(-1).reshape(-1)
    diag=dict(valid_candidate_count=usable.reshape(b,-1).sum(-1),
        shared_candidate_count=shared_candidate.reshape(b,-1).sum(-1),
        valid_member_count=valid_members.reshape(b,n).sum(-1),
        shared_member_count=shared_members.reshape(b,n).sum(-1),
        unique_evidence_nodes=_per_frame_sum(torch.ones(count,device=device,dtype=torch.long),node_frame,b),
        edge_count=_per_frame_sum(torch.ones_like(ef),ef,b),
        candidate_degree=candidate_degree.reshape(b,n,k),
        candidate_shared_mask=shared_candidate.reshape(b,n,k),
        evidence_candidate_degree=degree,evidence_distinct_member_degree=member_degree,
        shuffled_changed_edge_count=_per_frame_sum((eidx!=original_eidx).long(),ef,b),
        geometry_valid_candidate_count=geometry['valid'].reshape(b,-1).sum(-1),
        connected_member_count=usable.reshape(b,n,k).any(-1).sum(-1),
        graph_variant=variant,physical_psf_proxy=dict(scales),
        candidate_sharing_denominator='candidates with at least one valid evidence edge',
        member_sharing_denominator='members with at least one geometrically valid candidate',
        evidence_node_key='(frame,view,native_range_index,native_angle_index)',
        shuffled_attributes='Original physical edge attributes remain with source edges; destinations alone are rewired',
        degree_is_not_true_target_count=True)
    return dict(candidate_index=cidx,evidence_index=eidx,attributes=attributes,
                node_features=features,node_coords=coords,node_frame=node_frame,node_grid_ids=node_grid,
                candidate_owner=owners,evidence_candidate_degree=degree,evidence_member_degree=member_degree,
                usable=usable.reshape(b,n,k),diagnostics=diag)


def _mlp(a,h,out=None):
    return nn.Sequential(nn.Linear(a,h),nn.SiLU(),nn.Linear(h,h if out is None else out))


class MessageLayer(nn.Module):
    def __init__(self,h):
        super().__init__()
        self.ce=_mlp(2*h+6,h);self.ce_gate=nn.Linear(2*h+6,1)
        self.e_update=_mlp(3*h+2,h);self.e_norm=nn.LayerNorm(h)
        self.ec=_mlp(2*h+6,h);self.ec_gate=nn.Linear(2*h+6,1)
        self.c_update=_mlp(3*h,h);self.c_norm=nn.LayerNorm(h)
        self.m_update=_mlp(2*h,h);self.m_norm=nn.LayerNorm(h)
        self.mc=_mlp(2*h,h)

    def forward(self,candidate,evidence,member,graph,candidate_mass):
        ci,ei=graph['candidate_index'],graph['evidence_index'];edge=graph['attributes']
        inputs=torch.cat((candidate[ci],evidence[ei],edge),-1)
        weights=edge[:,3]*torch.sigmoid(self.ce_gate(inputs)[:,0])*candidate_mass[ci]
        messages=self.ce(inputs)*weights[:,None]
        total=_scatter_sum(messages,ei,len(evidence));mass=_scatter_sum(weights,ei,len(evidence)).clamp_min(1e-8)
        degree=graph['evidence_candidate_degree'];members=graph['evidence_member_degree']
        stats=torch.stack((degree.log1p(),members.log1p()),-1)
        evidence=self.e_norm(evidence+self.e_update(torch.cat((evidence,total/mass[:,None],total/(degree+1).sqrt()[:,None],stats),-1)))
        values=torch.cat((candidate[ci],evidence[ei],edge),-1)
        weights=edge[:,3]*torch.sigmoid(self.ec_gate(values)[:,0])
        feedback=_scatter_sum(self.ec(values)*weights[:,None],ci,len(candidate))
        normal=_scatter_sum(weights,ci,len(candidate)).clamp_min(1e-8)
        owner=graph['candidate_owner']
        candidate=self.c_norm(candidate+self.c_update(torch.cat((candidate,feedback/normal[:,None],member[owner]),-1)))
        pooled=_scatter_sum(candidate*candidate_mass[:,None],owner,len(member))
        member=self.m_norm(member+self.m_update(torch.cat((member,pooled),-1)))
        candidate=candidate+self.mc(torch.cat((candidate,member[owner]),-1))
        return candidate,evidence,member


class SharedEvidenceGraph(nn.Module):
    """Two-layer M→C↔shared-native-E→C→M evidence module."""
    def __init__(self,hidden=64,candidate_multipliers=None,rf=None,variant='shared'):
        super().__init__()
        if variant not in VARIANTS:raise ValueError('Unknown graph variant')
        self.variant=variant;self.hidden=hidden;self.scales=rf_scales(rf)
        values=[-3,-2,-1,-.5,0,.5,1,2,3.] if candidate_multipliers is None else candidate_multipliers
        self.register_buffer('candidate_multipliers',torch.tensor(values,dtype=torch.float32))
        self.member_encoder=_mlp(hidden+2,hidden)
        self.candidate_encoder=_mlp(3*hidden+14,hidden)
        self.evidence_encoder=_mlp(hidden+5,hidden)
        self.layers=nn.ModuleList([MessageLayer(hidden),MessageLayer(hidden)])
        self.candidate_score=nn.Linear(hidden,1)

    def forward(self,*,origin,direction,bounds,ray_valid,visual_mean,visual_std,visual_token,
                ra_map,re_map,range_axis,azimuth_axis,elevation_axis,ra_valid,re_valid,range_scale_m,sampler):
        b,n=visual_mean.shape;k=len(self.candidate_multipliers);h=self.hidden
        offsets=visual_std[...,None]*self.candidate_multipliers.to(visual_mean.dtype)
        depths=visual_mean[...,None]+offsets
        geometry=candidate_geometry(origin.reshape(-1,3),direction.reshape(-1,3),depths.reshape(-1,k),bounds.reshape(-1,2),ray_valid.reshape(-1))
        if geometry['positions'].requires_grad:
            valid=geometry['valid']
            geometry['positions'].register_hook(lambda grad:torch.where(valid[...,None],grad,torch.zeros_like(grad)))
        g={name:value.reshape((b,n,k)+(value.shape[2:] if name=='positions' else ())) for name,value in geometry.items()}
        ra,av=sampler(ra_map,ra_valid,g['ranges'],g['azimuth'],range_axis,azimuth_axis,g['valid'])
        re,ev=sampler(re_map,re_valid,g['ranges'],g['elevation'],range_axis,elevation_axis,g['valid'])
        graph=build_graph(g,ra_map,re_map,range_axis,azimuth_axis,elevation_axis,ra_valid,re_valid,n,self.scales,self.variant)
        usable=graph['usable'];scale=range_scale_m[:,None,None,None]
        angular=torch.stack((g['azimuth'].sin(),g['azimuth'].cos(),g['elevation'].sin(),g['elevation'].cos()),-1)
        scalars=torch.cat((offsets[...,None]/scale,(offsets/visual_std[...,None])[...,None],depths[...,None]/scale,
            g['ranges'][...,None]/scale,angular,av[...,None].to(visual_mean.dtype),ev[...,None].to(visual_mean.dtype),
            g['valid'][...,None].to(visual_mean.dtype),g['positions']/scale),-1)
        member=self.member_encoder(torch.cat((visual_token,visual_mean[...,None]/range_scale_m[:,None,None],
                                             visual_std[...,None]/range_scale_m[:,None,None]),-1)).reshape(b*n,h)
        inputs=torch.cat((ra,re,member.reshape(b,n,h)[:,:,None].expand(-1,-1,k,-1),scalars),-1)
        inputs=torch.where(usable[...,None],inputs,torch.zeros_like(inputs))
        candidate=self.candidate_encoder(inputs).reshape(-1,h)*usable.reshape(-1,1)
        evidence_coords=graph['node_coords'].clone()
        evidence_coords[:,0]=evidence_coords[:,0]/range_scale_m[graph['node_frame']]
        evidence=self.evidence_encoder(torch.cat((graph['node_features'],evidence_coords),-1))
        def distribution(values):
            logits=self.candidate_score(values).reshape(b,n,k)
            weights=torch.softmax(logits.masked_fill(~usable,torch.finfo(logits.dtype).min),-1)*usable
            return weights/weights.sum(-1,keepdim=True).clamp_min(torch.finfo(logits.dtype).eps)
        for layer in self.layers:
            probability=distribution(candidate).reshape(-1)
            candidate,evidence,member=layer(candidate,evidence,member,graph,probability)
            candidate=candidate*usable.reshape(-1,1)
        weights=distribution(candidate)
        pooled=(candidate.reshape(b,n,k,h)*weights[...,None]).sum(2)
        pooled=pooled*usable.any(-1)[...,None]
        diag=dict(graph['diagnostics'],query_rounds=1,candidate_depths_m=depths,candidate_offsets_m=offsets,
            geometry_valid=g['valid'],ra_valid=av,re_valid=ev,usable=usable,weights=weights,
            evidence_available=usable.any(-1),graph_architecture='member_candidate_evidence_v1')
        return pooled,diag
