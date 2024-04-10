# Set up unittests for the hard rasterizer

from functools import partial
import unittest
import torch
import slangtorch
import timeit
from torch.autograd import Function
import collections

class TestAABBIntersection(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.rasterizer2d = slangtorch.loadModule("unittest_bindings.slang")

    def test_aabb_intersection_1(self):
        aabb = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).type(torch.float).cuda()
        segment = torch.tensor([[-0.5, 0.5], [1.5, 0.5]]).type(torch.float).cuda()

        outValid = torch.zeros(1).type(torch.bool).cuda()
        outNear = torch.zeros((1,2)).type(torch.float).cuda()
        outFar = torch.zeros((1,2)).type(torch.float).cuda()

        self.rasterizer2d.aabb_intersection_direct_call(
            aabb=aabb,
            segment=segment,
            outValid=outValid,
            outNear=outNear,
            outFar=outFar).launchRaw(blockSize=(1,1,1), gridSize=(1,1,1))

        self.assertEqual(outValid[0], True)
        self.assertTrue(
            torch.allclose(outNear[0,:].cpu(), torch.tensor([0.0, 0.5])), f"outNear: {outNear[0,:]} not-close-to [0.0, 0.5]")
        self.assertTrue(
            torch.allclose(outFar[0,:].cpu(),  torch.tensor([1.0, 0.5])), f"outFar: {outFar[0,:]} not-close-to [1.0, 0.5]")
    
    def test_aabb_intersection_degenerate(self):
        aabb = torch.tensor([[0.0, 0.0], [1.0, 1.0]]).type(torch.float).cuda()
        segment = torch.tensor([[-1.0, 1.0], [2.0, 1.0]]).type(torch.float).cuda()

        outValid = torch.zeros(1).type(torch.bool).cuda()
        outNear = torch.zeros((1,2)).type(torch.float).cuda()
        outFar = torch.zeros((1,2)).type(torch.float).cuda()

        self.rasterizer2d.aabb_intersection_direct_call(
            aabb=aabb,
            segment=segment,
            outValid=outValid,
            outNear=outNear,
            outFar=outFar).launchRaw(blockSize=(1,1,1), gridSize=(1,1,1))

        self.assertEqual(outValid[0], False)

def normalize_along_last_axis(v):
    return v / torch.norm(v, dim=-1, keepdim=True)

class TestTriangle(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.rasterizer2d = slangtorch.loadModule("unittest_bindings.slang")

    def assertTensorsEqual(self, a, b, msg=None):
        self.assertTrue(torch.all(torch.eq(a, b)), msg = f"Tensors not close: {a} != {b}")
    
    def assertTensorsClose(self, a, b, msg=None):
        self.assertTrue(torch.allclose(a, b, atol=1e-6), msg = f"Tensors not close: {a} != {b}")

    def triangleSampleFromEdge(self, segment, aabb, sample1D):
        # Wrap inputs in torch tensors of the proper type & device.
        # Grab and return the outputs as a namedtuple.
        #
        outValid = torch.zeros(1).type(torch.bool).cuda()
        outPt = torch.zeros((1,2)).type(torch.float).cuda()
        outNormal = torch.zeros((1,2)).type(torch.float).cuda()

        segment = torch.tensor(segment).type(torch.float).cuda()
        aabb = torch.tensor(aabb).type(torch.float).cuda()
        sample1D = torch.tensor(sample1D).type(torch.float).cuda()

        self.rasterizer2d.triangle_sample_from_edge_direct_call(
            vertices=segment, 
            aabb=aabb, 
            sample1D=sample1D,
            outValid=outValid,
            outPt=outPt, 
            outNormal=outNormal).launchRaw(blockSize=(1,1,1), gridSize=(1,1,1))

        _tupletype = collections.namedtuple('EdgeSample', ['valid', 'pt', 'normal'])
        return _tupletype(outValid[0], outPt[0,:].cpu(), outNormal[0,:].cpu())

    def triangleSampleFromBoundary(self, vertices, aabb, sample1D):
        # Wrap inputs in torch tensors of the proper type & device.
        # Grab and return the outputs as a namedtuple.
        #
        outValid = torch.zeros(1).type(torch.bool).cuda()
        outPt = torch.zeros((1,2)).type(torch.float).cuda()
        outNormal = torch.zeros((1,2)).type(torch.float).cuda()

        vertices = torch.tensor(vertices).type(torch.float).cuda()
        aabb = torch.tensor(aabb).type(torch.float).cuda()
        sample1D = torch.tensor(sample1D).type(torch.float).cuda()

        self.rasterizer2d.triangle_sample_from_boundary_direct_call(
            vertices=vertices, 
            aabb=aabb, 
            sample1D=sample1D,
            outValid=outValid,
            outPt=outPt, 
            outNormal=outNormal).launchRaw(blockSize=(1,1,1), gridSize=(1,1,1))

        _tupletype = collections.namedtuple('BoundarySample', ['valid', 'pt', 'normal'])
        return _tupletype(outValid[0], outPt[0,:].cpu(), outNormal[0,:].cpu())

    def test_triangle_sample_from_flat_edge_intersected(self):
        # Flat edge intersected
        edgeSample = self.triangleSampleFromEdge(segment=[[-0.5, 0.5], [1.5, 0.5]], aabb=[[0.0, 0.0], [1.0, 1.0]], sample1D=0.5)
        self.assertEqual(edgeSample.valid, True)
        self.assertTensorsEqual(edgeSample.pt, torch.tensor([0.5, 0.5]))
        self.assertTensorsEqual(edgeSample.normal, torch.tensor([0.0, -1.0]))

        edgeSample = self.triangleSampleFromEdge(segment=[[-0.5, 0.5], [1.5, 0.5]], aabb=[[0.0, 0.0], [1.0, 1.0]], sample1D=0.0)
        self.assertEqual(edgeSample.valid, True)
        self.assertTensorsEqual(edgeSample.pt, torch.tensor([0.0, 0.5]))
        self.assertTensorsEqual(edgeSample.normal, torch.tensor([0.0, -1.0]))

        edgeSample = self.triangleSampleFromEdge(segment=[[-0.5, 0.5], [1.5, 0.5]], aabb=[[0.0, 0.0], [1.0, 1.0]], sample1D=1.0)
        self.assertEqual(edgeSample.valid, True)
        self.assertTensorsEqual(edgeSample.pt, torch.tensor([1.0, 0.5]))
        self.assertTensorsEqual(edgeSample.normal, torch.tensor([0.0, -1.0]))
    
    def test_triangle_sample_from_flat_edge_outside(self):
        # Rewritten in terms of the helper function.
        edgeSample = self.triangleSampleFromEdge(segment=[[-0.5, 2.0], [1.5, 2.0]], aabb=[[0.0, 0.0], [1.0, 1.0]], sample1D=0.5)
        self.assertEqual(edgeSample.valid, False)
    
    def test_triangle_sample_from_edge(self):
        # Edge completely outside.
        edgeSample = self.triangleSampleFromEdge(segment=[[-2, 0.5], [0.5, 2.0]], aabb=[[0.0, 0.0], [1.0, 1.0]], sample1D=0.5)
        self.assertEqual(edgeSample.valid, False)

        # Rewritten in terms of the helper function.
        edgeSample = self.triangleSampleFromEdge(segment=[[-0.5, 0.5], [1.5, 0.5]], aabb=[[0.0, 0.0], [1.0, 1.0]], sample1D=0.5)
        self.assertEqual(edgeSample.valid, True)
        self.assertTensorsEqual(edgeSample.pt, torch.tensor([0.5, 0.5]))
        self.assertTensorsEqual(edgeSample.normal, torch.tensor([0.0, -1.0]))

    def test_sample_from_triangle_boundary_single_segment(self):
        # Triangle completely outside.
        bSample = self.triangleSampleFromBoundary(vertices=[[2.0, 4.0], [4.0, 0.0], [4.0, 4.0]], aabb=[[0.0, 0.0], [1.0, 1.0]], sample1D=0.5)
        self.assertEqual(bSample.valid, False)

        # Single segment of triangle intersected. (Note CCW is front-facing in our implementation)
        singleSegmentTriangleSampler = partial(
            self.triangleSampleFromBoundary, vertices=[[-1, -1], [2.0, -1.0], [-1.0, 2.0]], aabb=[[0.0, 0.0], [1.0, 1.0]])
        
        bSample = singleSegmentTriangleSampler(sample1D=1.0/12.0)
        self.assertEqual(bSample.valid, True)
        self.assertTensorsClose(bSample.pt, torch.tensor([0.5, 0.5]))
        self.assertTensorsClose(bSample.normal, normalize_along_last_axis(torch.tensor([1.0, 1.0])))

        # Ensure that in the single segment case, the entire intersected segment is mapped.
        bSample = singleSegmentTriangleSampler(sample1D=0.0)
        self.assertEqual(bSample.valid, True)
        self.assertTensorsClose(bSample.pt, torch.tensor([1.0, 0.0]))
        self.assertTensorsClose(bSample.normal, normalize_along_last_axis(torch.tensor([1.0, 1.0])))

        bSample = singleSegmentTriangleSampler(sample1D=1.0)
        self.assertEqual(bSample.valid, True)
        self.assertTensorsClose(bSample.pt, torch.tensor([0.0, 1.0]))
        self.assertTensorsClose(bSample.normal, normalize_along_last_axis(torch.tensor([1.0, 1.0])))

    def test_sample_from_triangle_boundary_two_segments(self):
        # Two segments of triangle intersected.
        twoSegmentTriangleSampler = partial(
            self.triangleSampleFromBoundary, vertices=[[-1, -1], [3.0, -1.0], [1.0, 1.0]], aabb=[[0.0, 0.0], [2.0, 1.0]])
        
        bSample = twoSegmentTriangleSampler(sample1D=1.0/12.0)
        self.assertEqual(bSample.valid, True)
        self.assertTensorsClose(bSample.pt, torch.tensor([1.5, 0.5]))
        self.assertTensorsClose(bSample.normal, normalize_along_last_axis(torch.tensor([1.0, 1.0])))

        bSample = twoSegmentTriangleSampler(sample1D=1.0/12.0 + 1.0/6.0)
        self.assertEqual(bSample.valid, True)
        self.assertTensorsClose(bSample.pt, torch.tensor([0.5, 0.5]))
        self.assertTensorsClose(bSample.normal, normalize_along_last_axis(torch.tensor([-1.0, 1.0])))
    
    def test_sample_from_triangle_boundary_three_segments(self):
        # All three segments of triangle intersected.
        threeSegmentTriangleSampler = partial(
            self.triangleSampleFromBoundary, vertices=[[-1, -1], [3.0, -1.0], [1.0, 1.0]], aabb=[[0.0, -2.0], [2.0, 1.0]])
        
        bSample = threeSegmentTriangleSampler(sample1D=1.0/12.0)
        self.assertEqual(bSample.valid, True)
        self.assertTensorsClose(bSample.pt, torch.tensor([1.0, -1.0]))
        self.assertTensorsClose(bSample.normal, normalize_along_last_axis(torch.tensor([0.0, -1.0])))

        bSample = threeSegmentTriangleSampler(sample1D=1.0/12.0 + 1.0/6.0)
        self.assertEqual(bSample.valid, True)
        self.assertTensorsClose(bSample.pt, torch.tensor([1.5, 0.5]))
        self.assertTensorsClose(bSample.normal, normalize_along_last_axis(torch.tensor([1.0, 1.0])))

        bSample = threeSegmentTriangleSampler(sample1D=1.0/12.0 + 1.0/6.0 + 1.0/6.0)
        self.assertEqual(bSample.valid, True)
        self.assertTensorsClose(bSample.pt, torch.tensor([0.5, 0.5]))
        self.assertTensorsClose(bSample.normal, normalize_along_last_axis(torch.tensor([-1.0, 1.0])))


class TestDiffRenderPixel(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.rasterizer2d = slangtorch.loadModule("unittest_bindings.slang")

    def assertTensorsEqual(self, a, b, msg=None):
        self.assertTrue(torch.all(torch.eq(a, b)), msg = f"Tensors not close: {a} != {b}")
    
    def assertTensorsClose(self, a, b, msg=None):
        self.assertTrue(torch.allclose(a, b, atol=1e-6), msg = f"Tensors not close: {a} != {b}")
    
    def renderPixelFwdDiff(self, vertices, d_vertices, pixelID):
        # Wrap inputs in torch tensors of the proper type & device.
        # Grab and return the outputs as a namedtuple.
        #
        outColor = torch.zeros(3).type(torch.float).cuda()
        outDColor = torch.zeros(3).type(torch.float).cuda()
        rngState = torch.zeros(1).type(torch.int).cuda()

        vertices = torch.tensor(vertices).type(torch.float).cuda()
        d_vertices = torch.tensor(d_vertices).type(torch.float).cuda()
        pixelID = torch.tensor(pixelID).type(torch.float).cuda()

        self.rasterizer2d.render_pixel_direct_call.fwd(
            vertices=(vertices, d_vertices), 
            pixel=pixelID, 
            rngState=rngState,
            output=(outColor, outDColor)).launchRaw(blockSize=(1,1,1), gridSize=(1,1,1))

        return outDColor

    def test_render_pixel_fwd_diff(self):
        vertices = [[0.1,0.0], [-0.1,0.1], [-0.1,-0.1]]
        pixelID = [132, 123]
        d_vertices = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

        dColor = self.renderPixelFwdDiff(vertices, d_vertices, pixelID)

        self.assertTensorsClose(dColor.cpu(), torch.tensor([0.0000, -0.2625, -0.2625]))