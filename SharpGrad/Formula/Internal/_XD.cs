using ILGPU;
using ILGPU.Runtime;
using System;

namespace SharpGrad.Formula
{
    public interface IXD
    {
        abstract static int DimensionCount { get; }
    }

    internal readonly struct _1D : IXD
    {
        public static int DimensionCount => 1;
    }
    internal readonly struct _2D : IXD
    {
        public static int DimensionCount => 2;
    }
    internal readonly struct _3D : IXD
    {
        public static int DimensionCount => 3;
    }
}