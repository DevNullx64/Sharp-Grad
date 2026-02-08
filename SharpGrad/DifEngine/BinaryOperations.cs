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
                string methodName = $"{kind}Forward";
                MethodInfo genericMethod = CPU.DeviceCpu.FindGenericMethod(typeof(BinaryOperations), methodName)
                    ?? throw new InvalidOperationException($"Method {nameof(BinaryOperations)}.{methodName} not found.");
                method = genericMethod.MakeGenericMethod(typeof(TType));
                cacheBinaryKindForwardMethodInfos[(kind, typeof(TType))] = method;
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
                string methodName = $"{kind}BackwardLeft";
                MethodInfo genericMethod = CPU.DeviceCpu.FindGenericMethod(typeof(BinaryOperations), methodName)
                    ?? throw new InvalidOperationException($"Method {nameof(BinaryOperations)}.{methodName} not found.");
                method = genericMethod.MakeGenericMethod(typeof(TValue));
                cacheBinaryKindBackwardLeftMethodInfos[(kind, typeof(TValue), typeof(TGradient))] = method;
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
                string methodName = $"{kind}BackwardRight";
                MethodInfo genericMethod = CPU.DeviceCpu.FindGenericMethod(typeof(BinaryOperations), methodName)
                    ?? throw new InvalidOperationException($"Method {nameof(BinaryOperations)}.{methodName} not found.");
                method = genericMethod.MakeGenericMethod(typeof(TValue));
                cacheBinaryKindBackwardRightMethodInfos[(kind, typeof(TValue), typeof(TGradient))] = method;
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
        public static T AddBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AddBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SubtractForward<T>(T left, T right)
            where T : INumber<T>
            => left - right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SubtractBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SubtractBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => -gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MultiplyForward<T>(T left, T right)
            where T : INumber<T>
            => left * right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MultiplyBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => right * gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MultiplyBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => left * gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T DivideForward<T>(T left, T right)
            where T : INumber<T>
            => left / right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T DivideBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => gradOutput / right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T DivideBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => -left * gradOutput / (right * right);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T PowForward<T>(T left, T right)
            where T : INumber<T>, IPowerFunctions<T>
            => T.Pow(left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T PowBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IPowerFunctions<T>
            => right * T.Pow(left, right - T.One) * gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T PowBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IPowerFunctions<T>, ILogarithmicFunctions<T>
            => T.Log(left) * T.Pow(left, right) * gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T ModuloForward<T>(T left, T right)
            where T : INumber<T>
            => left % right;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T ModuloBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => gradOutput;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T ModuloBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>
            => -(left / right) * gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MaxForward<T>(T left, T right)
            where T : INumber<T>
            => T.Max(left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MaxBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IComparable<T>
            => left.CompareTo(right) switch
                {
                    < 0 => T.CreateTruncating(T.Zero),
                    > 0 => T.CreateTruncating(gradOutput),
                    _ => T.CreateTruncating(gradOutput / T.CreateChecked(2)),
                };
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MaxBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IComparable<T>
            => left.CompareTo(right) switch
                {
                    < 0 => T.CreateTruncating(gradOutput),
                    > 0 => T.CreateTruncating(T.Zero),
                    _ => T.CreateTruncating(gradOutput / T.CreateChecked(2)),
                };

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MinForward<T>(T left, T right)
            where T : INumber<T>
            => T.Min(left, right);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MinBackwardLeft<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IComparable<T>
            => left.CompareTo(right) switch
                {
                    < 0 => T.CreateTruncating(gradOutput),
                    > 0 => T.CreateTruncating(T.Zero),
                    _ => T.CreateTruncating(gradOutput / T.CreateChecked(2)),
                };
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MinBackwardRight<T>(T left, T right, T gradOutput)
            where T : INumber<T>, IComparable<T>
            => left.CompareTo(right) switch
            {
                < 0 => T.CreateTruncating(T.Zero),
                > 0 => T.CreateTruncating(gradOutput),
                _ => T.CreateTruncating(gradOutput / T.CreateChecked(2)),
            };
    }
}