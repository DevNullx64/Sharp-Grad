﻿using SharpGrad.DifEngine;
using System.Collections.Generic;
using System.Linq;

namespace SharpGrad
{
    public static class DimensionExtender
    {
        public static int Size(this IEnumerable<Dimension> @this)
            => @this.Aggregate(1, (acc, d) => acc * d.Size);

        public static bool IsScalar(this IEnumerable<Dimension> @this)
            => !@this.Any();
    }
}
