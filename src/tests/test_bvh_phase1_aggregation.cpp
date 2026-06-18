#include <mitsuba/render/bvh/bvh.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/thread.h>
#include <mitsuba/render/trimesh.h>
#include <mitsuba/render/bsdf.h>
#include <iostream>

MTS_NAMESPACE_BEGIN

// Helper function to create a simple triangle mesh
TriMesh* createTestTriangle() {
    // Triangle with vertices at (0,0,0), (1,0,0), (0,1,0)
    // Expected area: 0.5
    // Expected normal: (0,0,1)

    // Create mesh with 1 triangle, 3 vertices, with normals
    TriMesh *mesh = new TriMesh("test_triangle", 1, 3, true, false, false, false, false);

    // Set up triangle indices
    Triangle &tri = mesh->getTriangles()[0];
    tri.idx[0] = 0;
    tri.idx[1] = 1;
    tri.idx[2] = 2;

    // Set up vertex positions
    Point *vertices = mesh->getVertexPositions();
    vertices[0] = Point(0.0f, 0.0f, 0.0f);
    vertices[1] = Point(1.0f, 0.0f, 0.0f);
    vertices[2] = Point(0.0f, 1.0f, 0.0f);

    // Set up vertex normals
    Normal *normals = mesh->getVertexNormals();
    normals[0] = Normal(0.0f, 0.0f, 1.0f);
    normals[1] = Normal(0.0f, 0.0f, 1.0f);
    normals[2] = Normal(0.0f, 0.0f, 1.0f);

    // Create a simple diffuse BSDF with albedo 0.8
    Properties bsdfProps("diffuse");
    bsdfProps.setSpectrum("reflectance", Spectrum(0.8f));
    BSDF *bsdf = static_cast<BSDF *>(PluginManager::getInstance()->createObject(
        MTS_CLASS(BSDF), bsdfProps));
    bsdf->configure();
    mesh->addChild(bsdf);

    // Configure mesh (builds acceleration structures, etc.)
    mesh->configure();

    return mesh;
}

int main(int argc, char **argv) {
    // Minimal Mitsuba initialization
    Class::staticInitialization();
    Object::staticInitialization();
    PluginManager::staticInitialization();
    Thread::staticInitialization();
    FileStream::staticInitialization();
    Spectrum::staticInitialization();

    ref<FileResolver> fileResolver = new FileResolver();
    fileResolver->appendPath(".");
    Thread::getThread()->setFileResolver(fileResolver);

    std::cout << "=== BVH Phase 1 Tests ===" << std::endl;

    int passed = 0;
    int total = 0;

    // Test 1: Create empty BVHNodeInfo
    std::cout << "\n[Test 1] Create BVHNodeInfo..." << std::endl;
    total++;
    BVHNodeInfo info;
    std::cout << "  Initial area: " << info.surfaceArea << std::endl;
    std::cout << "  Initial albedo: " << info.getMeanDiffuseAlbedoLum() << std::endl;
    std::cout << "  Valid: " << (info.valid ? "true" : "false") << std::endl;

    if (!info.valid && info.surfaceArea == 0.0f) {
        std::cout << "  ✓ PASS" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    // Test 2: Create BVHNodeInfo manually
    std::cout << "\n[Test 2] Create BVHNodeInfo manually..." << std::endl;
    total++;
    BVHNodeInfo tri;
    tri.surfaceArea = 0.5f;
    tri.s_diffuse_albedo = Spectrum(0.8f) * tri.surfaceArea;
    tri.normalDistribution.setFromNormal(Vector(0, 0, 1));
    tri.centerOfMass = Point(0.333f, 0.333f, 0.0f);
    tri.valid = true;
    std::cout << "  Triangle area: " << tri.surfaceArea << std::endl;
    std::cout << "  Triangle albedo: " << tri.getMeanDiffuseAlbedoLum() << std::endl;

    if (tri.surfaceArea == 0.5f && std::abs(tri.getMeanDiffuseAlbedoLum() - 0.8f) < 1e-5f) {
        std::cout << "  ✓ PASS" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    // Test 3: Verify BVHNodeInfo fields are accessible
    std::cout << "\n[Test 3] Verify BVHNodeInfo fields..." << std::endl;
    total++;
    std::cout << "  Node area: " << tri.surfaceArea << std::endl;
    std::cout << "  Node albedo: " << tri.getMeanDiffuseAlbedoLum() << std::endl;
    std::cout << "  Valid: " << (tri.valid ? "true" : "false") << std::endl;

    if (tri.valid && tri.surfaceArea == 0.5f && std::abs(tri.getMeanDiffuseAlbedoLum() - 0.8f) < 1e-5f) {
        std::cout << "  ✓ PASS" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL" << std::endl;
    }

    // Test 4: Accumulate two nodes
    std::cout << "\n[Test 4] Accumulate two nodes..." << std::endl;
    total++;
    BVHNodeInfo tri1, tri2;
    tri1.surfaceArea = 0.5f;
    tri1.s_diffuse_albedo = Spectrum(0.3f) * tri1.surfaceArea;
    tri1.normalDistribution.setFromNormal(Vector(0, 0, 1));
    tri1.centerOfMass = Point(0.0f, 0.0f, 0.0f);
    tri1.valid = true;

    tri2.surfaceArea = 0.5f;
    tri2.s_diffuse_albedo = Spectrum(0.7f) * tri2.surfaceArea;
    tri2.normalDistribution.setFromNormal(Vector(0, 0, 1));
    tri2.centerOfMass = Point(1.0f, 0.0f, 0.0f);
    tri2.valid = true;

    BVHNodeInfo combined;
    combined.accumulate(tri1);
    combined.accumulate(tri2);

    std::cout << "  Combined area: " << combined.surfaceArea << " (expected: 1.0)" << std::endl;
    std::cout << "  Combined albedo: " << combined.getMeanDiffuseAlbedoLum() << " (expected: 0.5)" << std::endl;

    bool areaOk = std::abs(combined.surfaceArea - 1.0f) < 0.001f;
    bool albedoOk = std::abs(combined.getMeanDiffuseAlbedoLum() - 0.5f) < 0.001f;

    if (areaOk && albedoOk) {
        std::cout << "  ✓ PASS: Accumulation works correctly!" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL: Accumulation incorrect!" << std::endl;
    }

    // Test 4b: Accumulate nodes with different normals (test variance)
    std::cout << "\n[Test 4b] Accumulate nodes with different normals (variance)..." << std::endl;
    total++;
    BVHNodeInfo triA, triB;
    triA.surfaceArea = 1.0f;
    triA.s_diffuse_albedo = Spectrum(0.5f) * triA.surfaceArea;
    triA.normalDistribution.setFromNormal(Vector(0, 0, 1));      // Facing up
    triA.centerOfMass = Point(0.0f, 0.0f, 0.0f);
    triA.valid = true;

    triB.surfaceArea = 1.0f;
    triB.s_diffuse_albedo = Spectrum(0.5f) * triB.surfaceArea;
    triB.normalDistribution.setFromNormal(Vector(1, 0, 0));      // Facing right (90° difference)
    triB.centerOfMass = Point(1.0f, 0.0f, 0.0f);
    triB.valid = true;

    BVHNodeInfo combinedDifferent;
    combinedDifferent.accumulate(triA);
    combinedDifferent.accumulate(triB);

    std::cout << "  Combined variance (90° apart): " << combinedDifferent.normalDistribution.variance() << std::endl;
    Vector meanDiff = combinedDifferent.normalDistribution.meanDirection();
    std::cout << "  Combined mean normal: (" << meanDiff.x << ", "
              << meanDiff.y << ", " << meanDiff.z << ")" << std::endl;

    // Now combine two nodes with same normal - should have lower variance
    BVHNodeInfo triC, triD;
    triC.surfaceArea = 1.0f;
    triC.s_diffuse_albedo = Spectrum(0.5f) * triC.surfaceArea;
    triC.normalDistribution.setFromNormal(Vector(0, 0, 1));
    triC.centerOfMass = Point(0.0f, 0.0f, 0.0f);
    triC.valid = true;

    triD.surfaceArea = 1.0f;
    triD.s_diffuse_albedo = Spectrum(0.5f) * triD.surfaceArea;
    triD.normalDistribution.setFromNormal(Vector(0, 0, 1));      // Same direction
    triD.centerOfMass = Point(1.0f, 0.0f, 0.0f);
    triD.valid = true;

    BVHNodeInfo combinedSame;
    combinedSame.accumulate(triC);
    combinedSame.accumulate(triD);

    std::cout << "  Combined variance (same direction): " << combinedSame.normalDistribution.variance() << std::endl;

    // Different normals should produce higher variance
    bool varianceCorrect = combinedDifferent.normalDistribution.variance() > combinedSame.normalDistribution.variance() + 0.1f;

    if (varianceCorrect) {
        std::cout << "  ✓ PASS: Variance correctly accounts for normal differences!" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL: Variance should be higher when normals differ!" << std::endl;
    }

    // Test 5: Compute aggregate from real geometry
    std::cout << "\n[Test 5] Compute aggregate from real triangle geometry..." << std::endl;
    total++;
    ref<TriMesh> testMesh = createTestTriangle();

    BVHNodeInfo realTriInfo = computeTriangleInfo(testMesh.get(), 0, NULL);

    std::cout << "  Computed area: " << realTriInfo.surfaceArea << " (expected: 0.5)" << std::endl;
    Vector realMean = realTriInfo.normalDistribution.meanDirection();
    std::cout << "  Computed normal: (" << realMean.x << ", "
              << realMean.y << ", " << realMean.z
              << ") (expected: (0, 0, 1))" << std::endl;
    std::cout << "  Computed albedo: " << realTriInfo.getMeanDiffuseAlbedoLum() << " (expected: ~0.8)" << std::endl;

    bool areaCorrect = std::abs(realTriInfo.surfaceArea - 0.5f) < 0.001f;
    bool normalCorrect = std::abs(realMean.z - 1.0f) < 0.001f &&
                         std::abs(realMean.x) < 0.001f &&
                         std::abs(realMean.y) < 0.001f;
    bool albedoCorrect = std::abs(realTriInfo.getMeanDiffuseAlbedoLum() - 0.8f) < 0.1f; // Allow more tolerance for BSDF

    if (areaCorrect && normalCorrect && albedoCorrect) {
        std::cout << "  ✓ PASS: Real geometry computation works!" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL: Real geometry computation incorrect!" << std::endl;
        if (!areaCorrect) std::cout << "    - Area mismatch" << std::endl;
        if (!normalCorrect) std::cout << "    - Normal mismatch" << std::endl;
        if (!albedoCorrect) std::cout << "    - Albedo mismatch" << std::endl;
    }

    // Test 6: Accumulate real geometry aggregates
    std::cout << "\n[Test 6] Accumulate aggregates from real geometry..." << std::endl;
    total++;

    BVHNodeInfo accumulated;
    accumulated.accumulate(realTriInfo);
    accumulated.accumulate(realTriInfo); // Same triangle twice

    std::cout << "  Accumulated area: " << accumulated.surfaceArea << " (expected: 1.0)" << std::endl;
    std::cout << "  Accumulated albedo: " << accumulated.getMeanDiffuseAlbedoLum() << " (expected: ~0.8)" << std::endl;

    bool accAreaOk = std::abs(accumulated.surfaceArea - 1.0f) < 0.001f;
    bool accAlbedoOk = std::abs(accumulated.getMeanDiffuseAlbedoLum() - 0.8f) < 0.1f;

    if (accAreaOk && accAlbedoOk && accumulated.valid) {
        std::cout << "  ✓ PASS: Accumulation from real geometry works!" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL: Accumulation from real geometry incorrect!" << std::endl;
    }

    // Test 7: Importance function - node between surface and emitter
    std::cout << "\n[Test 7] Importance: Node between surface and emitter..." << std::endl;
    total++;

    // Setup: Surface at origin looking up, emitter at (0,0,10) looking down
    SurfaceSample xs(Point(0, 0, 0), Normal(0, 0, 1), NULL);
    EmitterSample xe(Point(0, 0, 10), Normal(0, 0, -1));

    // Node1: Directly between them at (0,0,5), facing up, high albedo
    BVHNodeInfo node1;
    node1.surfaceArea = 1.0f;
    node1.s_diffuse_albedo = Spectrum(0.8f) * node1.surfaceArea;
    node1.normalDistribution.setFromNormal(Vector(0, 0, 1));
    node1.centerOfMass = Point(0, 0, 5);
    node1.valid = true;
    AABB bounds1(Point(-0.5f, -0.5f, 4.5f), Point(0.5f, 0.5f, 5.5f));

    Float imp1 = computeNodeImportance(node1, bounds1, xs, xe);
    std::cout << "  Importance (node between): " << imp1 << std::endl;

    if (imp1 > 0.0f) {
        std::cout << "  ✓ PASS: Node between has positive importance" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL: Node between should have positive importance" << std::endl;
    }

    // Test 8: Importance function - node far away
    std::cout << "\n[Test 8] Importance: Node far from both points..." << std::endl;
    total++;

    // Node2: Far away at (100,0,5), same properties as node1
    BVHNodeInfo node2 = node1;
    node2.centerOfMass = Point(100, 0, 5);
    AABB bounds2(Point(99.5f, -0.5f, 4.5f), Point(100.5f, 0.5f, 5.5f));

    Float imp2 = computeNodeImportance(node2, bounds2, xs, xe);
    std::cout << "  Importance (far node): " << imp2 << std::endl;
    std::cout << "  Ratio imp1/imp2: " << (imp2 > 0.0f ? imp1/imp2 : -1.0f) << std::endl;

    // Far node should have much lower importance due to distance
    if (imp2 > 0.0f && imp1 > imp2 * 10.0f) {
        std::cout << "  ✓ PASS: Far node has much lower importance" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL: Far node should have much lower importance" << std::endl;
    }

    // Test 9: Importance function - node facing away
    std::cout << "\n[Test 9] Importance: Node facing away from emitter..." << std::endl;
    total++;

    // Node3: At (0,0,5) but facing down (away from emitter)
    BVHNodeInfo node3 = node1;
    node3.normalDistribution.setFromNormal(Vector(0, 0, -1)); // Facing opposite direction

    Float imp3 = computeNodeImportance(node3, bounds1, xs, xe);
    std::cout << "  Importance (facing away): " << imp3 << std::endl;
    std::cout << "  Ratio imp1/imp3: " << (imp3 > 0.0f ? imp1/imp3 : -1.0f) << std::endl;

    // Node facing away should have zero or very low importance
    if (imp3 < imp1 * 0.1f) {
        std::cout << "  ✓ PASS: Node facing away has very low importance" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL: Node facing away should have much lower importance" << std::endl;
    }

    // Test 10: Importance function - low albedo node
    std::cout << "\n[Test 10] Importance: Node with low albedo..." << std::endl;
    total++;

    // Node4: Same position as node1 but low albedo
    BVHNodeInfo node4 = node1;
    node4.s_diffuse_albedo = Spectrum(0.1f) * node4.surfaceArea;

    Float imp4 = computeNodeImportance(node4, bounds1, xs, xe);
    std::cout << "  Importance (low albedo): " << imp4 << std::endl;
    std::cout << "  Ratio imp1/imp4: " << (imp4 > 0.0f ? imp1/imp4 : -1.0f) << std::endl;

    // Low albedo should result in proportionally lower importance
    Float expectedRatio = node1.getMeanDiffuseAlbedoLum() / node4.getMeanDiffuseAlbedoLum(); // 0.8 / 0.1 = 8
    Float actualRatio = imp4 > 0.0f ? imp1 / imp4 : 0.0f;
    bool ratioCorrect = std::abs(actualRatio - expectedRatio) < 0.5f;

    if (ratioCorrect) {
        std::cout << "  ✓ PASS: Albedo scales importance correctly" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL: Albedo should scale importance proportionally" << std::endl;
    }

    // Test 11: Importance function - high variance node
    std::cout << "\n[Test 11] Importance: Node with high normal variance..." << std::endl;
    total++;

    // Node5: Same as node1 but with high normal variance
    BVHNodeInfo node5 = node1;
    // Create a "rough" distribution by reducing the length of E[n]
    // while keeping E[||n||^2] ~= 1 (identity/3 as a simple isotropic proxy).
    node5.normalDistribution.m1 = Vector(0.0f);
    node5.normalDistribution.m2 = Matrix3x3(
        1.0f/3.0f, 0.0f,     0.0f,
        0.0f,     1.0f/3.0f, 0.0f,
        0.0f,     0.0f,     1.0f/3.0f
    );

    Float imp5 = computeNodeImportance(node5, bounds1, xs, xe);
    std::cout << "  Importance (high variance): " << imp5 << std::endl;
    std::cout << "  Ratio imp1/imp5: " << (imp5 > 0.0f ? imp1/imp5 : -1.0f) << std::endl;

    // High variance should reduce importance (penalty for uncertain geometry)
    if (imp5 > 0.0f && imp1 > imp5 * 1.5f) {
        std::cout << "  ✓ PASS: High variance reduces importance" << std::endl;
        passed++;
    } else {
        std::cout << "  ✗ FAIL: High variance should reduce importance" << std::endl;
    }

    std::cout << "\n==========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;

    if (passed == total) {
        std::cout << "\n🎉 All tests PASSED! Phase 1 complete." << std::endl;
        std::cout << "Next step: Implement BVH tree construction and traversal!" << std::endl;
    } else {
        std::cout << "\n⚠️  Some tests failed. Please review the implementation." << std::endl;
    }

    return (passed == total) ? 0 : 1;
}

MTS_NAMESPACE_END

int main(int argc, char **argv) {
    return mitsuba::main(argc, argv);
}
