using SharpGrad.DifEngine.SyntaxBuilder.Operations;
using System;
using System.Collections.Generic;
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

        /// <summary>
        /// Executes the forward pass of a binary operation on the given left and right inputs and stores the result in the given output.
        /// </summary>
        /// <param name="kind">The kind of binary operation to execute.</param>
        /// <param name="left">The left input Value.</param>
        /// <param name="right">The right input Value.</param>
        /// <param name="output">The output Value.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the forward pass for different element types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="DeviceCpu.ExecuteBinaryForward"/> method is not found.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteBinaryForward(KindBinary kind, Value left, Value right, Value output)
        {
            // Check if a cached delegate exists for the operation kind and element type
            Type elementType = output.ElementType;
            if (!cacheExecuteBinaryForwards.TryGetValue((kind, elementType), out Action<KindBinary, Value, Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteBinaryForward<elementType>(BinaryKind, Value, Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteBinaryForward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(KindBinary), typeof(Value), typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteBinaryForward)} not found.");
                method = method.MakeGenericMethod(elementType);
                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<KindBinary, Value, Value, Value>>(this);
                // Cache the delegate for future use
                cacheExecuteBinaryForwards[(kind, elementType)] = action;
            }
            // Call the generic ExecuteBinaryForward<T> method
            action(kind, left, right, output);
        }

        /// <summary>
        /// Executes the forward pass of a binary operation for given left and right inputs and stores the result in the given output.
        /// </summary>
        /// <typeparam name="TType">The type of the elements in the left, right, and output Values. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <param name="kind">The kind of binary operation to execute.</param>
        /// <param name="untypedLeft">The left input <see cref="Value"/>.</param>
        /// <param name="untypedRight">The right input <see cref="Value"/>.</param>
        /// <param name="untypedOutput">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// <paramref name="untypedLeft"/>, <paramref name="untypedRight"/>, and <paramref name="untypedOutput"/> must be of type <see cref="Value{T}"/>.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the inputs or output are not of type <see cref="Value{T}"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="BinaryOperations"/>."[<see cref="KindBinary"/>]Forward" method is not found.</exception>
        private void ExecuteBinaryForward<TType>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where TType : struct, INumber<TType>
        {
            // Throw an exception if the inputs and output are not of the expected type
            TType[] left = untypedLeft.GetInitializedData<TType>();
            TType[] right = untypedRight.GetInitializedData<TType>();
            TType[] output = untypedOutput.GetOrInitializeData<TType>();

            // Get the appropriate method for the binary operation
            Func<TType, TType, TType> operation = BinaryOperations.GetKindForwardDelegate<TType>(kind);

            // Perform the binary operation
            Shape leftShape = untypedLeft.Shape;
            Shape rightShape = untypedRight.Shape;
            Shape outputShape = untypedOutput.Shape;
            int length = output.Length;
            if (leftShape == rightShape)
            {
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        output[iOutput] = operation(left[iOutput], right[iOutput]);
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        output[iOutput] = operation(left[iOutput], right[iOutput]);
                    });
                }
            }
            else
            {
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        int iLeft = leftShape.GetLinearIndex(iOutput, outputShape);
                        int iRight = rightShape.GetLinearIndex(iOutput, outputShape);
                        output[iOutput] = operation(left[iLeft], right[iRight]);
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        int iLeft = leftShape.GetLinearIndex(iOutput, outputShape);
                        int iRight = rightShape.GetLinearIndex(iOutput, outputShape);
                        output[iOutput] = operation(left[iLeft], right[iRight]);
                    });
                }
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

        /// <summary>
        /// Executes the backward pass of a binary operation on the given left and right inputs and output Value.
        /// </summary>
        /// <param name="kind">The kind of binary operation to execute backward pass for.</param>
        /// <param name="left">The left input <see cref="Value"/>.</param>
        /// <param name="right">The right input <see cref="Value"/>.</param>
        /// <param name="output">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the backward pass for different combinations of element types and gradient types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="DeviceCpu.ExecuteBinaryBackward"/> method is not found.</exception>
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
            // Check if a cached delegate exists for the operation kind and types
            Type valueType = output.ElementType;
            Type gradientType = output.Grad.ElementType;
            (KindBinary, Type, Type) key = (kind, valueType, gradientType);

            if (!cacheExecuteBinaryBackwardLeftOnlyDelegates.TryGetValue(key, out Action<KindBinary, Value, Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteBinaryBackwardLeftOnly<valueType, gradientType>(BinaryKind, Value, Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteBinaryBackwardLeftOnly),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(KindBinary), typeof(Value), typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteBinaryBackwardLeftOnly)} not found.");
                method = method.MakeGenericMethod(valueType, gradientType);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<KindBinary, Value, Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteBinaryBackwardLeftOnlyDelegates[key] = action;
            }
            // Call the generic ExecuteBinaryBackwardLeftOnly<T, G> method
            action(kind, left, right, output);
        }

        private void ExecuteBinaryBackwardLeftOnly<TType, TGrad>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where TType : struct, INumber<TType>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            TType[] left = untypedLeft.GetInitializedData<TType>();
            TGrad[] leftGrad = untypedLeft.GetOrInitializeGrad<TGrad>();
            TType[] right = untypedRight.GetInitializedData<TType>();
            TGrad[] outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

            Func<TType, TType, TGrad, TGrad> func = BinaryOperations.GetKindBackwardLeftDelegate<TType, TGrad>(kind);

            int length = outputGrad.Length;
            Shape leftShape = untypedLeft.Shape;
            Shape rightShape = untypedRight.Shape;
            Shape outputShape = untypedOutput.Shape;
            
            if (leftShape == rightShape)
            {
                // No broadcasting: direct accumulation
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        leftGrad[iOutput] += func(left[iOutput], right[iOutput], outputGrad[iOutput]);
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        leftGrad[iOutput] += func(left[iOutput], right[iOutput], outputGrad[iOutput]);
                    });
                }
            }
            else
            {
                // Broadcasting required: dimension-by-dimension reduction
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        int iLeft = leftShape.GetLinearIndex(iOutput, outputShape);
                        int iRight = rightShape.GetLinearIndex(iOutput, outputShape);
                        TGrad grad = func(left[iLeft], right[iRight], outputGrad[iOutput]);
                        leftGrad[iLeft] += grad;
                    }
                }
                else
                {
                    // Calculate gradients into temporary buffer
                    TGrad[] tempGrad = new TGrad[length];
                    
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        int iLeft = leftShape.GetLinearIndex(iOutput, outputShape);
                        int iRight = rightShape.GetLinearIndex(iOutput, outputShape);
                        tempGrad[iOutput] = func(left[iLeft], right[iRight], outputGrad[iOutput]);
                    });
                    
                    // Reduce dimension by dimension
                    ReduceBroadcastedGradient(tempGrad, outputShape, leftGrad, leftShape);
                }
            }
        }

        /// <summary>
        /// Reduces broadcasted gradient by accumulating over dimensions present in source but not in dest.
        /// Uses dimension-by-dimension reduction with temporary buffers for optimal performance.
        /// </summary>
        private void ReduceBroadcastedGradient<TGrad>(TGrad[] sourceGrad, Shape sourceShape, TGrad[] destGrad, Shape destShape)
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            List<Dimension> dimsToReduce = [];

            for (int iSource = 0; iSource < sourceShape.Rank; iSource++)
            {
                Dimension sourceDim = sourceShape[iSource];
                if (destShape.IndexOf(sourceDim) < 0)
                {
                    dimsToReduce.Add(sourceDim);
                }
            }

            if (dimsToReduce.Count == 0)
            {
                Parallel.For(0, Math.Min(sourceGrad.Length, destGrad.Length), _parallelOptions, i =>
                {
                    destGrad[i] += sourceGrad[i];
                });
                return;
            }

            TGrad[] currentSource = sourceGrad;
            Shape currentShape = sourceShape;

            for (int dimIdx = dimsToReduce.Count - 1; dimIdx >= 0; dimIdx--)
            {
                Dimension dim = dimsToReduce[dimIdx];
                int dimIndexInShape = currentShape.IndexOf(dim);
                int dimSize = dim.Size;
                int dimStride = currentShape.GetStride(dimIndexInShape);
                Shape reducedShape = currentShape.Remove(dim);
                int reducedLength = currentSource.Length / dimSize;
                bool isFinal = dimIdx == 0;
                TGrad[] currentDest = isFinal ? destGrad : new TGrad[reducedLength];

                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iDest = reducedLength - 1; iDest >= 0; iDest--)
                    {
                        int iSourceBase = currentShape.GetLinearIndex(iDest, reducedShape);
                        TGrad accumulator = TGrad.Zero;
                        int iSourceEnd = iSourceBase + dimSize * dimStride;
                        for (int iSource = iSourceBase; iSource < iSourceEnd; iSource += dimStride)
                        {
                            accumulator += currentSource[iSource];
                        }

                        if (isFinal)
                        {
                            currentDest[iDest] += accumulator;
                        }
                        else
                        {
                            currentDest[iDest] = accumulator;
                        }
                    }
                }
                else
                {
                    Parallel.For(0, reducedLength, _parallelOptions, iDest =>
                    {
                        int iSourceBase = currentShape.GetLinearIndex(iDest, reducedShape);
                        TGrad accumulator = TGrad.Zero;
                        int iSourceEnd = iSourceBase + dimSize * dimStride;
                        for (int iSource = iSourceBase; iSource < iSourceEnd; iSource += dimStride)
                        {
                            accumulator += currentSource[iSource];
                        }

                        if (isFinal)
                        {
                            currentDest[iDest] += accumulator;
                        }
                        else
                        {
                            currentDest[iDest] = accumulator;
                        }
                    });
                }

                currentSource = currentDest;
                currentShape = reducedShape;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteBinaryBackwardRightOnly(KindBinary kind, Value left, Value right, Value output)
        {
            // Check if a cached delegate exists for the operation kind and types
            Type valueType = output.ElementType;
            Type gradientType = output.Grad.ElementType;
            (KindBinary, Type, Type) key = (kind, valueType, gradientType);

            if (!cacheExecuteBinaryBackwardRightOnlyDelegates.TryGetValue(key, out Action<KindBinary, Value, Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteBinaryBackwardRightOnly<valueType, gradientType>(BinaryKind, Value, Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteBinaryBackwardRightOnly),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(KindBinary), typeof(Value), typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteBinaryBackwardRightOnly)} not found.");
                method = method.MakeGenericMethod(valueType, gradientType);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<KindBinary, Value, Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteBinaryBackwardRightOnlyDelegates[key] = action;
            }
            // Call the generic ExecuteBinaryBackwardRightOnly<T, G> method
            action(kind, left, right, output);
        }

        private void ExecuteBinaryBackwardRightOnly<TType, TGrad>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where TType : struct, INumber<TType>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            TType[] left = untypedLeft.GetInitializedData<TType>();
            TType[] right = untypedRight.GetInitializedData<TType>();
            TGrad[] rightGrad = untypedRight.GetOrInitializeGrad<TGrad>();
            TGrad[] outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

            Func<TType, TType, TGrad, TGrad> func = BinaryOperations.GetKindBackwardRightDelegate<TType, TGrad>(kind);

            int length = outputGrad.Length;
            Shape leftShape = untypedLeft.Shape;
            Shape rightShape = untypedRight.Shape;
            Shape outputShape = untypedOutput.Shape;
            
            if (leftShape == rightShape)
            {
                // No broadcasting: direct accumulation
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        rightGrad[iOutput] += func(left[iOutput], right[iOutput], outputGrad[iOutput]);
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        rightGrad[iOutput] += func(left[iOutput], right[iOutput], outputGrad[iOutput]);
                    });
                }
            }
            else
            {
                // Broadcasting required
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        int iLeft = leftShape.GetLinearIndex(iOutput, outputShape);
                        int iRight = rightShape.GetLinearIndex(iOutput, outputShape);
                        TGrad grad = func(left[iLeft], right[iRight], outputGrad[iOutput]);
                        rightGrad[iRight] += grad;
                    }
                }
                else
                {
                    TGrad[] rightGradTemp = new TGrad[length];

                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        int iLeft = leftShape.GetLinearIndex(iOutput, outputShape);
                        int iRight = rightShape.GetLinearIndex(iOutput, outputShape);
                        rightGradTemp[iOutput] = func(left[iLeft], right[iRight], outputGrad[iOutput]);
                    });

                    // Use shared reduction method
                    ReduceBroadcastedGradient(rightGradTemp, outputShape, rightGrad, rightShape);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteBinaryBackwardLeftAndRight(KindBinary kind, Value left, Value right, Value output)
        {
            // Check if a cached delegate exists for the operation kind and types
            Type valueType = output.ElementType;
            Type gradientType = output.Grad.ElementType;
            (KindBinary, Type, Type) key = (kind, valueType, gradientType);

            if (!cacheExecuteBinaryBackwardLeftAndRightDelegates.TryGetValue(key, out Action<KindBinary, Value, Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteBinaryBackwardLeftAndRight<valueType, gradientType>(BinaryKind, Value, Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteBinaryBackwardLeftAndRight),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(KindBinary), typeof(Value), typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteBinaryBackwardLeftAndRight)} not found.");
                method = method.MakeGenericMethod(valueType, gradientType);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<KindBinary, Value, Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteBinaryBackwardLeftAndRightDelegates[key] = action;
            }
            // Call the generic ExecuteBinaryBackwardLeftAndRight<T, G> method
            action(kind, left, right, output);
        }

        private void ExecuteBinaryBackwardLeftAndRight<TType, TGrad>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where TType : struct, INumber<TType>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            TType[] left = untypedLeft.GetInitializedData<TType>();
            TGrad[] leftGrad = untypedLeft.GetOrInitializeGrad<TGrad>();
            TType[] right = untypedRight.GetInitializedData<TType>();
            TGrad[] rightGrad = untypedRight.GetOrInitializeGrad<TGrad>();
            TGrad[] outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

            Func<TType, TType, TGrad, TGrad> funcLeft = BinaryOperations.GetKindBackwardLeftDelegate<TType, TGrad>(kind);
            Func<TType, TType, TGrad, TGrad> funcRight = BinaryOperations.GetKindBackwardRightDelegate<TType, TGrad>(kind);

            int length = outputGrad.Length;
            Shape leftShape = untypedLeft.Shape;
            Shape rightShape = untypedRight.Shape;
            Shape outputShape = untypedOutput.Shape;
            
            if (leftShape == rightShape)
            {
                // No broadcasting: direct accumulation
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        TGrad gradLeft = funcLeft(left[iOutput], right[iOutput], outputGrad[iOutput]);
                        TGrad gradRight = funcRight(left[iOutput], right[iOutput], outputGrad[iOutput]);
                        leftGrad[iOutput] += gradLeft;
                        rightGrad[iOutput] += gradRight;
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        TGrad gradLeft = funcLeft(left[iOutput], right[iOutput], outputGrad[iOutput]);
                        TGrad gradRight = funcRight(left[iOutput], right[iOutput], outputGrad[iOutput]);
                        leftGrad[iOutput] += gradLeft;
                        rightGrad[iOutput] += gradRight;
                    });
                }
            }
            else
            {
                bool leftNeedsBroadcast = leftShape != outputShape;
                bool rightNeedsBroadcast = rightShape != outputShape;
                
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        int iLeft = leftShape.GetLinearIndex(iOutput, outputShape);
                        int iRight = rightShape.GetLinearIndex(iOutput, outputShape);

                        TGrad gradLeft = funcLeft(left[iLeft], right[iRight], outputGrad[iOutput]);
                        TGrad gradRight = funcRight(left[iLeft], right[iRight], outputGrad[iOutput]);
                        leftGrad[iLeft] += gradLeft;
                        rightGrad[iRight] += gradRight;
                    }
                }
                else
                {
                    TGrad[] leftGradTemp = new TGrad[length];
                    TGrad[] rightGradTemp = new TGrad[length];
                    
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        int iLeft = leftShape.GetLinearIndex(iOutput, outputShape);
                        int iRight = rightShape.GetLinearIndex(iOutput, outputShape);

                        TGrad gradLeft = funcLeft(left[iLeft], right[iRight], outputGrad[iOutput]);
                        TGrad gradRight = funcRight(left[iLeft], right[iRight], outputGrad[iOutput]);
                        leftGradTemp[iOutput] = gradLeft;
                        rightGradTemp[iOutput] = gradRight;
                    });
                    
                    // Use shared reduction method for both gradients
                    if (leftNeedsBroadcast)
                    {
                        ReduceBroadcastedGradient(leftGradTemp, outputShape, leftGrad, leftShape);
                    }
                    else
                    {
                        // Direct 1-to-1 mapping
                        Parallel.For(0, leftGrad.Length, _parallelOptions, iLeft =>
                        {
                            leftGrad[iLeft] += leftGradTemp[iLeft];
                        });
                    }
                    
                    if (rightNeedsBroadcast)
                    {
                        ReduceBroadcastedGradient(rightGradTemp, outputShape, rightGrad, rightShape);
                    }
                    else
                    {
                        // Direct 1-to-1 mapping
                        Parallel.For(0, rightGrad.Length, _parallelOptions, iRight =>
                        {
                            rightGrad[iRight] += rightGradTemp[iRight];
                        });
                    }
                }
            }
        }
    }


}
