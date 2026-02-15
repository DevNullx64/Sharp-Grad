using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace SharpGrad.DifEngine.SyntaxBuilder.CPU
{
    public partial class DeviceCpu
    {
        // Cache for the ExecuteBinaryForward delegates
        private readonly Dictionary<
            (KindBinary kind, Type Type),
            Action<KindBinary, Value, Value, Value>> cacheExecuteBinaryForwards = [];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteBinaryForward(KindBinary kind, Value left, Value right, Value output)
        {
            Type elementType = output.ElementType;
            if (!cacheExecuteBinaryForwards.TryGetValue((kind, elementType), out Action<KindBinary, Value, Value, Value>? action))
            {
                MethodInfo method = GetGenericMethod(
                    nameof(ExecuteBinaryForward),
                    1,
                    typeof(KindBinary), typeof(Value), typeof(Value), typeof(Value))
                    ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteBinaryForward)} not found.");
                method = method.MakeGenericMethod(elementType);
                action = method.CreateDelegate<Action<KindBinary, Value, Value, Value>>(this);
                cacheExecuteBinaryForwards[(kind, elementType)] = action;
            }
            action(kind, left, right, output);
        }

        private void ExecuteBinaryForward<TType>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where TType : struct, INumber<TType>
        {
            untypedOutput.InitializeData();

            Func<TType, TType, TType> operation = BinaryOperations.GetKindForwardDelegate<TType>(kind);

            Shape leftShape = untypedLeft.Shape;
            Shape rightShape = untypedRight.Shape;
            Shape outputShape = untypedOutput.Shape;
            int length = outputShape.Size;
            
            if (leftShape == rightShape)
            {
                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TType> output = untypedOutput.GetInitializedDataSpan<TType>();

                    for (int iOutput = range.Item1; iOutput < range.Item2; iOutput++)
                    {
                        output[iOutput] = operation(left[iOutput], right[iOutput]);
                    }
                });
            }
            else
            {
                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TType> output = untypedOutput.GetInitializedDataSpan<TType>();

                    for (int iOutput = range.Item1; iOutput < range.Item2; iOutput++)
                    {
                        int iLeft = leftShape.GetLinearIndex(iOutput, outputShape, true);
                        int iRight = rightShape.GetLinearIndex(iOutput, outputShape, true);
                        output[iOutput] = operation(left[iLeft], right[iRight]);
                    }
                });
            }
        }

        // Cache for the ExecuteBinaryBackward delegates
        private readonly Dictionary<
            (KindBinary kind, Type Value, Type Gradient),
            Action<KindBinary, Value, Value, Value>> cacheExecuteBinaryBackwardLeftOnlyDelegates = [];
        private readonly Dictionary<
            (KindBinary kind, Type Value, Type Gradient),
            Action<KindBinary, Value, Value, Value>> cacheExecuteBinaryBackwardRightOnlyDelegates = [];
        private readonly Dictionary<
            (KindBinary kind, Type Value, Type Gradient),
            Action<KindBinary, Value, Value, Value>> cacheExecuteBinaryBackwardLeftAndRightDelegates = [];

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteBinaryBackward(KindBinary kind, Value left, Value right, Value output)
        {
            if (left.IsGradiable && right.IsGradiable)
            {
                ExecuteBinaryBackwardLeftAndRight(kind, left, right, output);
            }
            else if (left.IsGradiable)
            {
                ExecuteBinaryBackwardLeftOnly(kind, left, right, output);
            }
            else if (right.IsGradiable)
            {
                ExecuteBinaryBackwardRightOnly(kind, left, right, output);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteBinaryBackwardLeftOnly(KindBinary kind, Value left, Value right, Value output)
        {
            Type valueType = output.ElementType;
            Type gradientType = output.Grad.ElementType;
            (KindBinary, Type, Type) key = (kind, valueType, gradientType);

            if (!cacheExecuteBinaryBackwardLeftOnlyDelegates.TryGetValue(key, out Action<KindBinary, Value, Value, Value>? action))
            {
                MethodInfo method = GetGenericMethod(
                    nameof(ExecuteBinaryBackwardLeftOnly),
                    2,
                    typeof(KindBinary), typeof(Value), typeof(Value), typeof(Value))
                    ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteBinaryBackwardLeftOnly)} not found.");
                method = method.MakeGenericMethod(valueType, gradientType);
                action = method.CreateDelegate<Action<KindBinary, Value, Value, Value>>(this);
                cacheExecuteBinaryBackwardLeftOnlyDelegates[key] = action;
            }
            action(kind, left, right, output);
        }

        private void ExecuteBinaryBackwardLeftOnly<TType, TGrad>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where TType : struct, INumber<TType>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            untypedLeft.InitializeGrad<TGrad>();

            Func<TType, TType, TGrad, TGrad> func = BinaryOperations.GetKindBackwardLeftDelegate<TType, TGrad>(kind);

            Shape leftShape = untypedLeft.Shape;
            Shape rightShape = untypedRight.Shape;
            Shape outputShape = untypedOutput.Shape;
            int length = untypedOutput.Shape.Size;

            if (leftShape == rightShape)
            {
                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TGrad> leftGrad = untypedLeft.GetInitializedGrad<TGrad>();
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TGrad> outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        leftGrad[i] += func(left[i], right[i], outputGrad[i]);
                    }
                });
            }
            else if (leftShape == outputShape)
            {
                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TGrad> leftGrad = untypedLeft.GetInitializedGrad<TGrad>();
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TGrad> outputGrad = untypedOutput.GetInitializedGrad<TGrad>();
                    
                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        int iRight = rightShape.GetLinearIndex(i, outputShape, true);
                        leftGrad[i] += func(left[i], right[iRight], outputGrad[i]);
                    }
                });
            }
            else
            {
                TGrad[] toReduceLeftGrad = new TGrad[length];

                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TGrad> leftGrad = toReduceLeftGrad;
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TGrad> outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        int iLeft = leftShape.GetLinearIndex(i, outputShape, true);
                        int iRight = rightShape.GetLinearIndex(i, outputShape, true);
                        leftGrad[i] += func(left[iLeft], right[iRight], outputGrad[i]);
                    }
                });

                ReduceBroadcastedGradientToShape(toReduceLeftGrad, outputShape, leftShape, untypedLeft);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteBinaryBackwardRightOnly(KindBinary kind, Value left, Value right, Value output)
        {
            Type valueType = output.ElementType;
            Type gradientType = output.Grad.ElementType;
            (KindBinary, Type, Type) key = (kind, valueType, gradientType);

            if (!cacheExecuteBinaryBackwardRightOnlyDelegates.TryGetValue(key, out Action<KindBinary, Value, Value, Value>? action))
            {
                MethodInfo method = GetGenericMethod(
                    nameof(ExecuteBinaryBackwardRightOnly),
                    2,
                    typeof(KindBinary), typeof(Value), typeof(Value), typeof(Value))
                    ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteBinaryBackwardRightOnly)} not found.");
                method = method.MakeGenericMethod(valueType, gradientType);
                action = method.CreateDelegate<Action<KindBinary, Value, Value, Value>>(this);
                cacheExecuteBinaryBackwardRightOnlyDelegates[key] = action;
            }
            action(kind, left, right, output);
        }

        private void ExecuteBinaryBackwardRightOnly<TType, TGrad>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where TType : struct, INumber<TType>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            untypedRight.InitializeGrad<TGrad>();

            Func<TType, TType, TGrad, TGrad> func = BinaryOperations.GetKindBackwardRightDelegate<TType, TGrad>(kind);

            Shape leftShape = untypedLeft.Shape;
            Shape rightShape = untypedRight.Shape;
            Shape outputShape = untypedOutput.Shape;
            int length = outputShape.Size;

            if (leftShape == rightShape)
            {
                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TGrad> rightGrad = untypedRight.GetInitializedGrad<TGrad>();
                    Span<TGrad> outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        rightGrad[i] += func(left[i], right[i], outputGrad[i]);
                    }
                });
            }
            else if (rightShape == outputShape)
            {
                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TGrad> rightGrad = untypedRight.GetInitializedGrad<TGrad>();
                    Span<TGrad> outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        int iLeft = leftShape.GetLinearIndex(i, outputShape, true);
                        rightGrad[i] += func(left[iLeft], right[i], outputGrad[i]);
                    }
                });
            }
            else
            {
                TGrad[] toReduceRightGrad = new TGrad[length];

                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TGrad> rightGrad = toReduceRightGrad;
                    Span<TGrad> outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        int iLeft = leftShape.GetLinearIndex(i, outputShape, true);
                        int iRight = rightShape.GetLinearIndex(i, outputShape, true);
                        rightGrad[i] += func(left[iLeft], right[iRight], outputGrad[i]);
                    }
                });

                ReduceBroadcastedGradientToShape(toReduceRightGrad, outputShape, rightShape, untypedRight);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteBinaryBackwardLeftAndRight(KindBinary kind, Value left, Value right, Value output)
        {
            Type valueType = output.ElementType;
            Type gradientType = output.Grad.ElementType;
            (KindBinary, Type, Type) key = (kind, valueType, gradientType);

            if (!cacheExecuteBinaryBackwardLeftAndRightDelegates.TryGetValue(key, out Action<KindBinary, Value, Value, Value>? action))
            {
                MethodInfo method = GetGenericMethod(
                    nameof(ExecuteBinaryBackwardLeftAndRight),
                    2,
                    typeof(KindBinary), typeof(Value), typeof(Value), typeof(Value))
                    ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteBinaryBackwardLeftAndRight)} not found.");
                method = method.MakeGenericMethod(valueType, gradientType);
                action = method.CreateDelegate<Action<KindBinary, Value, Value, Value>>(this);
                cacheExecuteBinaryBackwardLeftAndRightDelegates[key] = action;
            }
            action(kind, left, right, output);
        }

        private void ExecuteBinaryBackwardLeftAndRight<TType, TGrad>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where TType : struct, INumber<TType>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            untypedLeft.InitializeGrad<TGrad>();
            untypedRight.InitializeGrad<TGrad>();

            Func<TType, TType, TGrad, TGrad> funcLeft = BinaryOperations.GetKindBackwardLeftDelegate<TType, TGrad>(kind);
            Func<TType, TType, TGrad, TGrad> funcRight = BinaryOperations.GetKindBackwardRightDelegate<TType, TGrad>(kind);

            Shape leftShape = untypedLeft.Shape;
            Shape rightShape = untypedRight.Shape;
            Shape outputShape = untypedOutput.Shape;
            int length = outputShape.Size;

            if (leftShape == rightShape)
            {
                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TGrad> leftGrad = untypedLeft.GetInitializedGrad<TGrad>();
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TGrad> rightGrad = untypedRight.GetInitializedGrad<TGrad>();
                    Span<TGrad> outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        leftGrad[i] += funcLeft(left[i], right[i], outputGrad[i]);
                        rightGrad[i] += funcRight(left[i], right[i], outputGrad[i]);
                    }
                });
            }
            else if (leftShape == outputShape)
            {
                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TGrad> leftGrad = untypedLeft.GetInitializedGrad<TGrad>();
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TGrad> rightGrad = untypedRight.GetInitializedGrad<TGrad>();
                    Span<TGrad> outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        int iRight = rightShape.GetLinearIndex(i, outputShape, true);
                        leftGrad[i] += funcLeft(left[i], right[iRight], outputGrad[i]);
                        rightGrad[iRight] += funcRight(left[i], right[iRight], outputGrad[i]);
                    }
                });
            }
            else if (rightShape == outputShape)
            {
                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TGrad> leftGrad = untypedLeft.GetInitializedGrad<TGrad>();
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TGrad> rightGrad = untypedRight.GetInitializedGrad<TGrad>();
                    Span<TGrad> outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        int iLeft = leftShape.GetLinearIndex(i, outputShape, true);
                        leftGrad[iLeft] += funcLeft(left[iLeft], right[i], outputGrad[i]);
                        rightGrad[i] += funcRight(left[iLeft], right[i], outputGrad[i]);
                    }
                });
            }
            else
            {
                TGrad[] toReduceLeftGrad = new TGrad[length];
                TGrad[] toReduceRightGrad = new TGrad[length];

                ParallelFor(0, length, range =>
                {
                    Span<TType> left = untypedLeft.GetInitializedDataSpan<TType>();
                    Span<TGrad> leftGrad = toReduceLeftGrad;
                    Span<TType> right = untypedRight.GetInitializedDataSpan<TType>();
                    Span<TGrad> rightGrad = toReduceRightGrad;
                    Span<TGrad> outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

                    for (int i = range.Item1; i < range.Item2; i++)
                    {
                        int iLeft = leftShape.GetLinearIndex(i, outputShape, true);
                        int iRight = rightShape.GetLinearIndex(i, outputShape, true);
                        leftGrad[i] += funcLeft(left[iLeft], right[iRight], outputGrad[i]);
                        rightGrad[i] += funcRight(left[iLeft], right[iRight], outputGrad[i]);
                    }
                });

                ReduceBroadcastedGradientToShape(toReduceLeftGrad, outputShape, leftShape, untypedLeft);
                ReduceBroadcastedGradientToShape(toReduceRightGrad, outputShape, rightShape, untypedRight);
            }
        }
    }
}