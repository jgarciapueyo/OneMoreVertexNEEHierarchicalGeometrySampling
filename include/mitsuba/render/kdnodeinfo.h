/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.
    Extensions for geometry-aware sampling by Additional Vertex Integrator.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#if !defined(__MITSUBA_RENDER_KDNODEINFO_H_)
#define __MITSUBA_RENDER_KDNODEINFO_H_

#include <mitsuba/mitsuba.h>
#include <mitsuba/core/vector.h>

MTS_NAMESPACE_BEGIN

/**
 * \brief Stores aggregated geometric information for a KD-tree node.
 *
 * This structure holds pre-computed properties (diffuse albedo, surface area,
 * normal distribution) for geometry-aware sampling. Built as a parallel array
 * indexed by KD-tree node index.
 *
 * \ingroup librender
 */
struct MTS_EXPORT_RENDER KDNodeInfo {
    /**
     * \brief SGGX-inspired Normal Distribution Function parameters.
     *
     * Stores a simplified isotropic representation of the normal distribution
     * within a node. The mean normal represents the principal direction, and
     * variance captures the spread of normals around the mean.
     *
     * For future extension: could store full 3x3 covariance matrix for
     * anisotropic normal distributions (full SGGX).
     */
    struct SGGX {
        /// Principal (mean) normal direction, normalized
        Normal meanNormal;

        /// Variance/spread of normals around the mean (isotropic approximation)
        /// 0 = perfectly flat surface, higher = more curved/varied normals
        Float variance;

        /// Default constructor
        inline SGGX() : meanNormal(0, 1, 0), variance(1.0f) {}

        /// Construct with specific values
        inline SGGX(const Normal &n, Float var)
            : meanNormal(n), variance(var) {}
    };

    /// Average diffuse reflectance (luminance) across shapes in this node.
    /// Computed from BSDF::getDiffuseReflectance() averaged over the surface.
    Float diffuseAlbedo;

    /// Total surface area of all primitives in this node (and subtree for internal nodes)
    Float surfaceArea;

    /// Normal distribution function parameters
    SGGX normalDistribution;

    /// True if this node contains valid geometry data
    bool valid;

    /// Default constructor - initializes to invalid state
    inline KDNodeInfo()
        : diffuseAlbedo(0.5f), surfaceArea(0.0f), valid(false) {}

    /// Construct with explicit values
    inline KDNodeInfo(Float albedo, Float area, const Normal &meanN, Float variance)
        : diffuseAlbedo(albedo), surfaceArea(area),
          normalDistribution(meanN, variance), valid(true) {}

    /**
     * \brief Compute the sampling weight for this node.
     *
     * Used during adaptive KD-tree traversal to probabilistically
     * select branches. Higher weight = more likely to be sampled.
     *
     * \return diffuseAlbedo * surfaceArea (importance weight)
     */
    inline Float getSamplingWeight() const {
        return diffuseAlbedo * surfaceArea;
    }

    /**
     * \brief Combine two child node infos into a parent (area-weighted).
     *
     * Used during bottom-up aggregation when building the node info array.
     *
     * \param left  Left child node info
     * \param right Right child node info
     * \return Combined parent node info
     */
    static inline KDNodeInfo combine(const KDNodeInfo &left, const KDNodeInfo &right) {
        KDNodeInfo result;

        // Total area is sum of children
        result.surfaceArea = left.surfaceArea + right.surfaceArea;

        if (result.surfaceArea > 0) {
            Float leftWeight = left.surfaceArea / result.surfaceArea;
            Float rightWeight = right.surfaceArea / result.surfaceArea;

            // Area-weighted average of albedo
            result.diffuseAlbedo = left.diffuseAlbedo * leftWeight
                                 + right.diffuseAlbedo * rightWeight;

            // Area-weighted average of mean normal
            Vector combinedNormal = Vector(left.normalDistribution.meanNormal) * leftWeight
                                  + Vector(right.normalDistribution.meanNormal) * rightWeight;
            Float combinedLength = combinedNormal.length();
            if (combinedLength > 0) {
                result.normalDistribution.meanNormal = Normal(combinedNormal / combinedLength);
            } else {
                result.normalDistribution.meanNormal = Normal(0, 1, 0);
            }

            // Combine variances: weighted average of variances + variance of means
            // This captures both the inherent variance and the spread due to different means
            Float varianceOfMeans = 0.0f;
            if (left.valid && right.valid) {
                // Approximate variance of means as (1 - dot(n1, n2)) which is 0 for same normals
                Float dotProduct = dot(left.normalDistribution.meanNormal,
                                       right.normalDistribution.meanNormal);
                varianceOfMeans = std::max(0.0f, 1.0f - dotProduct);
            }

            result.normalDistribution.variance =
                left.normalDistribution.variance * leftWeight +
                right.normalDistribution.variance * rightWeight +
                varianceOfMeans;

            result.valid = left.valid || right.valid;
        } else {
            // No area - use defaults
            result.diffuseAlbedo = 0.5f;
            result.normalDistribution = SGGX();
            result.valid = false;
        }

        return result;
    }

    /// Return a string representation
    std::string toString() const {
        std::ostringstream oss;
        oss << "KDNodeInfo[" << std::endl
            << "  diffuseAlbedo = " << diffuseAlbedo << "," << std::endl
            << "  surfaceArea = " << surfaceArea << "," << std::endl
            << "  meanNormal = " << normalDistribution.meanNormal.toString() << "," << std::endl
            << "  variance = " << normalDistribution.variance << "," << std::endl
            << "  valid = " << (valid ? "true" : "false") << std::endl
            << "]";
        return oss.str();
    }
};

MTS_NAMESPACE_END

#endif /* __MITSUBA_RENDER_KDNODEINFO_H_ */
