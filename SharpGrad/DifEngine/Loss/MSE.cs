using System.Numerics;

namespace SharpGrad.DifEngine.Loss
{
    public static partial class Loss
    {
        public static Value<TType> MSE<TType>(this Value<TType> Y, Value<TType> Y_hat, Dimension batch)
            where TType : struct, IBinaryFloatingPointIeee754<TType>
        {
            var diff = Y - Y_hat;
            var squaredDiff = diff * diff;
            var sumSquaredDiff = VMath.Sum(squaredDiff, batch);
            var mse = sumSquaredDiff / TType.CreateChecked(batch.Size);
            return mse;
        }
    }
}
