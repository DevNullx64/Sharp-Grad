using System;
using System.Collections.Generic;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;

namespace SharpGrad.DifEngine.SyntaxBuilder
{
    public static class UnaryOperations
    {
        private static readonly Dictionary<
            (Type Type, KindGraphNode kind),
            MethodInfo> cacheUnaryKindForwardMethodInfos = [];

        public static MethodInfo GetKindForwardMethodInfo<T>(KindGraphNode kind)
            where T : INumber<T>
        {
            if (!cacheUnaryKindForwardMethodInfos.TryGetValue((typeof(T), kind), out MethodInfo? method))
            {
                string methodName = $"{kind}Forward";
                method = typeof(UnaryOperations).GetMethod(
                    methodName,
                    BindingFlags.Public | BindingFlags.Static,
                    [typeof(T), typeof(T)]
                ) ?? throw new InvalidOperationException($"Method {nameof(UnaryOperations)}.{methodName} not found.");
                method = method.MakeGenericMethod(typeof(T));
                cacheUnaryKindForwardMethodInfos[(typeof(T), kind)] = method;
            }
            return method;
        }

        public static Func<TType, TType> GetKindForwardDelegate<TType>(KindGraphNode kind)
            where TType : INumber<TType>
        {
            var methodInfo = GetKindForwardMethodInfo<TType>(kind);
            return methodInfo.CreateDelegate<Func<TType, TType>>();
        }

        // Cache for the ExecuteBackward delegates MethodInfos
        private static readonly Dictionary<
            (KindGraphNode kind, Type Value, Type Gradient),
            MethodInfo> cacheExecuteBackwardMethodsInfos = [];

        public static MethodInfo GetKindBackwardMethodInfo<TValue, TGradient>(KindGraphNode kind)
            where TValue : INumber<TValue>
            where TGradient : IFloatingPoint<TGradient>
        {
            if (!cacheExecuteBackwardMethodsInfos.TryGetValue((kind, typeof(TValue), typeof(TGradient)), out MethodInfo? method))
            {
                string methodName = $"{kind}Backward";
                method = typeof(UnaryOperations).GetMethod(
                    methodName,
                    BindingFlags.Public | BindingFlags.Static,
                    [typeof(TValue), typeof(TValue), typeof(TGradient)]
                ) ?? throw new InvalidOperationException($"Method {nameof(UnaryOperations)}.{methodName} not found.");
                method = method.MakeGenericMethod(typeof(TValue), typeof(TGradient));
                cacheExecuteBackwardMethodsInfos[(kind, typeof(TValue), typeof(TGradient))] = method;
            }
            return method;
        }

        public static Func<TValue, TValue, TGradient, TGradient> GetKindBackwardDelegate<TValue, TGradient>(KindGraphNode kind)
            where TValue : INumber<TValue>
            where TGradient : IFloatingPoint<TGradient>
        {
            var methodInfo = GetKindBackwardMethodInfo<TValue, TGradient>(kind);
            return methodInfo.CreateDelegate<Func<TValue, TValue, TGradient, TGradient>>();
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T NegateForward<T>(T input)
        where T : INumber<T>
        => -input;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G NegateBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>
            => -gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T ReciprocalForward<T>(T input)
            where T : INumber<T>
            => T.One / input;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G ReciprocalBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>
        {
            var inputSquared = G.CreateTruncating(input);
            inputSquared *= inputSquared;
            return -gradOutput / inputSquared;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SqrtForward<T>(T input)
            where T : INumber<T>, IRootFunctions<T>
            => T.Sqrt(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G SqrtBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, IRootFunctions<G>
        {
            var two = G.One + G.One;
            var resultConverted = G.CreateTruncating(output);
            return gradOutput / (two * resultConverted);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SqForward<T>(T input)
            where T : INumber<T>
            => input * input;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G SqBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>
        {
            var two = G.One + G.One;
            var inputConverted = G.CreateTruncating(input);
            return two * inputConverted * gradOutput;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T ExpForward<T>(T input)
            where T : INumber<T>, IExponentialFunctions<T>
            => T.Exp(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G ExpBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, IExponentialFunctions<G>
        {
            var resultConverted = G.CreateTruncating(output);
            return resultConverted * gradOutput;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T LogForward<T>(T input)
            where T : INumber<T>, ILogarithmicFunctions<T>
            => T.Log(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G LogBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, ILogarithmicFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return gradOutput / inputConverted;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T Log10Forward<T>(T input)
            where T : INumber<T>, ILogarithmicFunctions<T>
            => T.Log10(input);
        private static double Log_10 = Math.Log(10.0);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G Log10Backward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, ILogarithmicFunctions<G>
        {
            var log10 = G.CreateTruncating(Log_10);
            return gradOutput / (G.CreateTruncating(input) * log10);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SinForward<T>(T input)
            where T : INumber<T>, ITrigonometricFunctions<T>
            => T.Sin(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G SinBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, ITrigonometricFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return gradOutput * G.Cos(inputConverted);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AsinForward<T>(T input)
            where T : INumber<T>, ITrigonometricFunctions<T>
            => T.Asin(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G AsinBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, IRootFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return gradOutput / G.Sqrt(G.One - inputConverted * inputConverted);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T CosForward<T>(T input)
            where T : INumber<T>, ITrigonometricFunctions<T>
            => T.Cos(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G CosBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, ITrigonometricFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return -gradOutput * G.Sin(inputConverted);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AcosForward<T>(T input)
            where T : INumber<T>, ITrigonometricFunctions<T>
            => T.Acos(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G AcosBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, IRootFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return -gradOutput / G.Sqrt(G.One - inputConverted * inputConverted);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T TanForward<T>(T input)
            where T : INumber<T>, ITrigonometricFunctions<T>
            => T.Tan(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G TanBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, ITrigonometricFunctions<G>
        {
            var cosInput = G.Cos(G.CreateTruncating(input));
            return gradOutput / (cosInput * cosInput);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AtanForward<T>(T input)
            where T : INumber<T>, ITrigonometricFunctions<T>
            => T.Atan(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G AtanBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, ITrigonometricFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return gradOutput / (G.One + inputConverted * inputConverted);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SinhForward<T>(T input)
            where T : INumber<T>, IHyperbolicFunctions<T>
            => T.Sinh(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G SinhBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, IHyperbolicFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return gradOutput * G.Cosh(inputConverted);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AsinhForward<T>(T input)
            where T : INumber<T>, IHyperbolicFunctions<T>
            => T.Asinh(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G AsinhBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, IRootFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return gradOutput / G.Sqrt(inputConverted * inputConverted + G.One);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T CoshForward<T>(T input)
            where T : INumber<T>, IHyperbolicFunctions<T>
            => T.Cosh(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G CoshBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, IHyperbolicFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return gradOutput * G.Sinh(inputConverted);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AcoshForward<T>(T input)
            where T : INumber<T>, IHyperbolicFunctions<T>
            => T.Acosh(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G AcoshBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, IRootFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return gradOutput / G.Sqrt(inputConverted * inputConverted - G.One);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T TanhForward<T>(T input)
            where T : INumber<T>, IHyperbolicFunctions<T>
            => T.Tanh(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G TanhBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, IHyperbolicFunctions<G>
        {
            var resultConverted = G.CreateTruncating(output);
            return gradOutput * (G.One - resultConverted * resultConverted);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AtanhForward<T>(T input)
            where T : INumber<T>, IHyperbolicFunctions<T>
            => T.Atanh(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G AtanhBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>
            where G : IFloatingPoint<G>, IHyperbolicFunctions<G>
        {
            var inputConverted = G.CreateTruncating(input);
            return gradOutput / (G.One - inputConverted * inputConverted);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T FloorForward<T>(T input)
            where T : INumber<T>, IFloatingPoint<T>
            => T.Floor(input);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T CeilForward<T>(T input)
            where T : INumber<T>, IFloatingPoint<T>
            => T.Ceiling(input);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T TruncForward<T>(T input)
            where T : INumber<T>, IFloatingPoint<T>
            => T.Truncate(input);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T RoundForward<T>(T input)
            where T : INumber<T>, IFloatingPoint<T>
            => T.Round(input);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T AbsForward<T>(T input)
            where T : INumber<T>
            => T.Abs(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G AbsBackward<T, G>(T input, T output, G gradOutput)
            where T : INumber<T>, ISignedNumber<T>
            where G : IFloatingPoint<G>
            => G.CreateTruncating(T.Sign(input)) * gradOutput;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T SignForward<T>(T input)
            where T : INumber<T>
            => T.CreateTruncating(T.Sign(input));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static TTo CastForward<TFrom, TTo>(TFrom input)
            where TFrom : INumber<TFrom>
            where TTo : INumber<TTo>
            => TTo.CreateTruncating(input);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static G CastBackward<TFrom, TTo, G>(TFrom input, TTo output, G gradOutput)
            where TFrom : INumber<TFrom>
            where TTo : INumber<TTo>
            where G : IFloatingPoint<G>
            => gradOutput;
    }
}