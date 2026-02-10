using System;
using System.Collections.Generic;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    public static class BinaryOperations
    {
        // Cache for the ExecuteBinaryForward MethodInfos
        private static readonly Dictionary<
            (KindBinary kind, Type Type),
            MethodInfo> cacheBinaryKindForwardMethodInfos = [];
        private static readonly object ForwardCacheLock = new();

        /// <summary>
        /// Gets the MethodInfo for the forward binary operation corresponding to the given KindBinary and type TType.
        /// </summary>
        /// <typeparam name="TType">The numeric type for which to get the MethodInfo.</typeparam>
        /// <param name="kind">The KindBinary representing the binary operation.</param>
        /// <returns>The MethodInfo for the forward binary operation.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the corresponding method is not found in the BinaryOperations class.</exception>
        public static MethodInfo GetKindForwardMethodInfos<TType>(KindBinary kind)
                where TType : struct, INumber<TType>
        {
            if (!cacheBinaryKindForwardMethodInfos.TryGetValue((kind, typeof(TType)), out MethodInfo? method))
            {
                lock (ForwardCacheLock)
                {
                    if (!cacheBinaryKindForwardMethodInfos.TryGetValue((kind, typeof(TType)), out method))
                    {
                        String methodName = $"{kind}Forward";
                        MethodInfo genericMethod = CPU.DeviceCpu.FindGenericMethod(typeof(BinaryOperations), methodName)
                            ?? throw new InvalidOperationException($"Method {nameof(BinaryOperations)}.{methodName} not found.");
                        method = genericMethod.MakeGenericMethod(typeof(TType));
                        cacheBinaryKindForwardMethodInfos[(kind, typeof(TType))] = method;
                    }
                }
            }
            return method;
        }

        public static Func<T, T, T> GetKindForwardDelegate<T>(KindBinary kind)
                where T : struct, INumber<T>
        {
            MethodInfo methodInfo = GetKindForwardMethodInfos<T>(kind);
            return methodInfo.CreateDelegate<Func<T, T, T>>();
        }

        // Cache for the ExecuteBinaryBackwardLeft MethodInfos
        private static readonly Dictionary<
            (KindBinary kind, Type Value, Type Gradient),
            MethodInfo> cacheBinaryKindBackwardLeftMethodInfos = [];
        private static readonly object BackwardLeftCacheLock = new();

        /// <summary>
        /// Gets the MethodInfo for the backward binary operation corresponding to the given KindBinary, value type TValue, and gradient type TGradient.
        /// </summary>
        /// <typeparam name="TValue">The numeric type of the value for which to get the MethodInfo.</typeparam>
        /// <typeparam name="TGradient">The numeric type of the gradient for which to get the MethodInfo.</typeparam>
        /// <param name="kind">The KindBinary representing the binary operation.</param>
        /// <returns>The MethodInfo for the backward binary operation.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the corresponding method is not found in the BinaryOperations class.</exception>
        public static MethodInfo GetKindBackwardLeftMethodInfos<TValue, TGradient>(KindBinary kind)
                where TValue : struct, INumber<TValue>
                where TGradient : struct, INumber<TGradient>
        {
            if (!cacheBinaryKindBackwardLeftMethodInfos.TryGetValue((kind, typeof(TValue), typeof(TGradient)), out MethodInfo? method))
            {
                lock (BackwardLeftCacheLock)
                {
                    if (!cacheBinaryKindBackwardLeftMethodInfos.TryGetValue((kind, typeof(TValue), typeof(TGradient)), out method))
                    {
                        String methodName = $"{kind}BackwardLeft";
                        MethodInfo genericMethod = CPU.DeviceCpu.FindGenericMethod(typeof(BinaryOperations), methodName)
                            ?? throw new InvalidOperationException($"Method {nameof(BinaryOperations)}.{methodName} not found.");
                        method = genericMethod.MakeGenericMethod(typeof(TValue), typeof(TGradient));
                        cacheBinaryKindBackwardLeftMethodInfos[(kind, typeof(TValue), typeof(TGradient))] = method;
                    }
                }
            }
            return method;
        }

        public static Func<T, T, G, G> GetKindBackwardLeftDelegate<T, G>(KindBinary kind)
                where T : struct, INumber<T>
                where G : struct, INumber<G>
        {
            MethodInfo methodInfo = GetKindBackwardLeftMethodInfos<T, G>(kind);
            return methodInfo.CreateDelegate<Func<T, T, G, G>>();
        }

        // Cache for the ExecuteBinaryBackwardRight MethodInfos
        private static readonly Dictionary<
            (KindBinary kind, Type Value, Type Gradient),
            MethodInfo> cacheBinaryKindBackwardRightMethodInfos = [];
        private static readonly object BackwardRightCacheLock = new();

        /// <summary>
        /// Gets the MethodInfo for the backward binary operation corresponding to the given KindBinary, value type TValue, and gradient type TGradient.
        /// </summary>
        /// <typeparam name="TValue">The numeric type of the value for which to get the MethodInfo.</typeparam>
        /// <typeparam name="TGradient">The numeric type of the gradient for which to get the MethodInfo.</typeparam>
        /// <param name="kind">The KindBinary representing the binary operation.</param>
        /// <returns>The MethodInfo for the backward binary operation.</returns>
        /// <exception cref="InvalidOperationException">Thrown if the corresponding method is not found in the BinaryOperations class.</exception>
        public static MethodInfo GetKindBackwardRightMethodInfos<TValue, TGradient>(KindBinary kind)
                where TValue : struct, INumber<TValue>
                where TGradient : struct, INumber<TGradient>
        {
            if (!cacheBinaryKindBackwardRightMethodInfos.TryGetValue((kind, typeof(TValue), typeof(TGradient)), out MethodInfo? method))
            {
                lock (BackwardRightCacheLock)
                {
                    if (!cacheBinaryKindBackwardRightMethodInfos.TryGetValue((kind, typeof(TValue), typeof(TGradient)), out method))
                    {
                        String methodName = $"{kind}BackwardRight";
                        MethodInfo genericMethod = CPU.DeviceCpu.FindGenericMethod(typeof(BinaryOperations), methodName)
                            ?? throw new InvalidOperationException($"Method {nameof(BinaryOperations)}.{methodName} not found.");
                        method = genericMethod.MakeGenericMethod(typeof(TValue), typeof(TGradient));
                        cacheBinaryKindBackwardRightMethodInfos[(kind, typeof(TValue), typeof(TGradient))] = method;
                    }
                }
            }
            return method;
        }

        public static Func<T, T, G, G> GetKindBackwardRightDelegate<T, G>(KindBinary kind)
                where T : struct, INumber<T>
                where G : struct, INumber<G>
        {
            MethodInfo methodInfo = GetKindBackwardRightMethodInfos<T, G>(kind);
            return methodInfo.CreateDelegate<Func<T, T, G, G>>();
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AddForward<T>(T left, T right)
            where T : INumber<T>
            => left + right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad AddBackwardLeft<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad AddBackwardRight<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SubtractForward<T>(T left, T right)
            where T : INumber<T>
            => left - right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad SubtractBackwardLeft<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad SubtractBackwardRight<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => -gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MultiplyForward<T>(T left, T right)
            where T : INumber<T>
            => left * right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad MultiplyBackwardLeft<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => TGrad.CreateTruncating(right) * gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad MultiplyBackwardRight<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => TGrad.CreateTruncating(left) * gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T DivideForward<T>(T left, T right)
            where T : INumber<T>
            => left / right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad DivideBackwardLeft<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => gradOutput / TGrad.CreateTruncating(right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad DivideBackwardRight<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => -TGrad.CreateTruncating(left) * gradOutput / (TGrad.CreateTruncating(right) * TGrad.CreateTruncating(right));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T PowForward<T>(T left, T right)
            where T : INumber<T>, IPowerFunctions<T>
            => T.Pow(left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad PowBackwardLeft<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>, IPowerFunctions<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>, IPowerFunctions<TGrad>
            => TGrad.CreateTruncating(right) * TGrad.Pow(TGrad.CreateTruncating(left), TGrad.CreateTruncating(right) - TGrad.One) * gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad PowBackwardRight<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>, IPowerFunctions<TValue>, ILogarithmicFunctions<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>, IPowerFunctions<TGrad>, ILogarithmicFunctions<TGrad>
            => TGrad.Log(TGrad.CreateTruncating(left)) * TGrad.Pow(TGrad.CreateTruncating(left), TGrad.CreateTruncating(right)) * gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T PowerForward<T>(T left, T right)
            where T : INumber<T>, IPowerFunctions<T>
            => PowForward(left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad PowerBackwardLeft<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>, IPowerFunctions<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>, IPowerFunctions<TGrad>
            => PowBackwardLeft(left, right, gradOutput);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad PowerBackwardRight<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>, IPowerFunctions<TValue>, ILogarithmicFunctions<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>, IPowerFunctions<TGrad>, ILogarithmicFunctions<TGrad>
            => PowBackwardRight(left, right, gradOutput);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T ModuloForward<T>(T left, T right)
            where T : INumber<T>
            => left % right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad ModuloBackwardLeft<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad ModuloBackwardRight<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => -(TGrad.CreateTruncating(left) / TGrad.CreateTruncating(right)) * gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MaxForward<T>(T left, T right)
            where T : INumber<T>
            => T.Max(left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad MaxBackwardLeft<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>, IComparable<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => left.CompareTo(right) switch
                {
                    < 0 => TGrad.Zero,
                    > 0 => gradOutput,
                    _ => gradOutput / TGrad.CreateTruncating(2),
                };
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad MaxBackwardRight<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>, IComparable<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => left.CompareTo(right) switch
                {
                    < 0 => gradOutput,
                    > 0 => TGrad.Zero,
                    _ => gradOutput / TGrad.CreateTruncating(2),
                };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MinForward<T>(T left, T right)
            where T : INumber<T>
            => T.Min(left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad MinBackwardLeft<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>, IComparable<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => left.CompareTo(right) switch
                {
                    < 0 => gradOutput,
                    > 0 => TGrad.Zero,
                    _ => gradOutput / TGrad.CreateTruncating(2),
                };
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TGrad MinBackwardRight<TValue, TGrad>(TValue left, TValue right, TGrad gradOutput)
            where TValue : struct, INumber<TValue>, IComparable<TValue>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
            => left.CompareTo(right) switch
            {
                < 0 => TGrad.Zero,
                > 0 => gradOutput,
                _ => gradOutput / TGrad.CreateTruncating(2),
            };
    }
}