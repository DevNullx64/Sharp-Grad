using SharpGrad;
using System.Numerics;
using System.Threading;

namespace TestProject
{
    public static class Common
    {
        private static Random rnd = new();
        private static int dimensionCounter;

        public static Dimension CreateDimension(string baseName, int size)
            => new($"{baseName}_{Interlocked.Increment(ref dimensionCounter)}", size);

        public static T Random<T>(T min, T max)
            where T : INumber<T>, IMultiplyOperators<T, T, T>, ISubtractionOperators<T, T, T>
            => T.CreateTruncating(double.CreateTruncating(max - min) * rnd.NextDouble()) + min;

        public static T Random<T>()
            where T : INumber<T>, IMultiplyOperators<T, T, T>, ISubtractionOperators<T, T, T>
            => T.CreateTruncating(rnd.NextDouble());
    }
}