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
            Value<TType> leftValue = untypedLeft.As<TType>();
            DataBuffer<TType> leftData = leftValue.GetOrInitializeDataBuffer();
            TType[] left = leftData.GetInitializedDataArray();

            Value<TType> rightValue = untypedRight.As<TType>();
            DataBuffer<TType> rightData = rightValue.GetOrInitializeDataBuffer();
            TType[] right = rightData.GetInitializedDataArray();

            Value<TType> outputValue = untypedOutput.As<TType>();
            DataBuffer<TType> outputData = outputValue.GetOrInitializeDataBuffer();
            TType[] output = outputData.GetInitializedDataArray();

            // Get the appropriate method for the binary operation
            Func<TType, TType, TType> operation = BinaryOperations.GetKindForwardDelegate<TType>(kind);

            // Perform the binary operation
            int length = output.Length;
            if (leftValue.Shape == rightValue.Shape)
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
                        int iLeft = leftValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        int iRight = rightValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        output[iOutput] = operation(left[iLeft], right[iRight]);
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        int iLeft = leftValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        int iRight = rightValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
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

        private void ExecuteBinaryBackwardLeftOnly<T, G>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where T : struct, INumber<T>
            where G : struct, IFloatingPointIeee754<G>
        {
            Value<T> leftValue = untypedLeft.As<T>();
            T[] left = leftValue.data.GetInitializedDataArray();
            DataBuffer<G> typedGradLeft = leftValue.GetOrInitializeGradBuffer<G>();
            G[] leftGrad = typedGradLeft.GetInitializedDataArray();

            Value<T> rightValue = untypedRight.As<T>();
            DataBuffer<T> rightData = rightValue.GetOrInitializeDataBuffer();
            T[] right = rightData.GetInitializedDataArray();

            Value<T> outputValue = untypedOutput.As<T>();
            DataBuffer<G> outputGrad = outputValue.GetOrInitializeGradBuffer<G>();
            G[] outputGradValue = outputGrad.GetInitializedDataArray();

            Func<T, T, G, G> func = BinaryOperations.GetKindBackwardLeftDelegate<T, G>(kind);

            int length = outputGradValue.Length;
            if (leftValue.Shape == rightValue.Shape)
            {
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        leftGrad[iOutput] += func(left[iOutput], right[iOutput], outputGradValue[iOutput]);
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        leftGrad[iOutput] += func(left[iOutput], right[iOutput], outputGradValue[iOutput]);
                    });
                }
            }
            else
            {
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        int iLeft = leftValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        int iRight = rightValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        G grad = func(left[iLeft], right[iRight], outputGradValue[iOutput]);
                        leftGrad[iLeft] += grad;
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        int iLeft = leftValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        int iRight = rightValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        G grad = func(left[iLeft], right[iRight], outputGradValue[iOutput]);
                        lock (leftGrad)
                        {
                            leftGrad[iLeft] += grad;
                        }
                    });
                }
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

        private void ExecuteBinaryBackwardRightOnly<T, G>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where T : struct, INumber<T>
            where G : struct, IFloatingPointIeee754<G>
        {
            Value<T> leftValue = untypedLeft.As<T>();
            DataBuffer<T> typedLeftData = leftValue.GetOrInitializeDataBuffer();
            T[] left = typedLeftData.GetInitializedDataArray();

            Value<T> rightValue = untypedRight.As<T>();
            DataBuffer<T> typedRightData = rightValue.GetOrInitializeDataBuffer();
            T[] right = typedRightData.GetInitializedDataArray();

            DataBuffer<G> typedGradRight = rightValue.GetOrInitializeGradBuffer<G>();
            G[] rightGrad = typedGradRight.GetInitializedDataArray();

            Value<T> outputValue = untypedOutput.As<T>();
            DataBuffer<G> outputGrad = outputValue.GetOrInitializeGradBuffer<G>();
            G[] outputGradValue = outputGrad.GetInitializedDataArray();

            Func<T, T, G, G> func = BinaryOperations.GetKindBackwardRightDelegate<T, G>(kind);

            int length = outputGradValue.Length;
            if (leftValue.Shape == rightValue.Shape)
            {
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        rightGrad[iOutput] += func(left[iOutput], right[iOutput], outputGradValue[iOutput]);
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        rightGrad[iOutput] += func(left[iOutput], right[iOutput], outputGradValue[iOutput]);
                    });
                }
            }
            else
            {
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        int iLeft = leftValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        int iRight = rightValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        G grad = func(left[iLeft], right[iRight], outputGradValue[iOutput]);
                        rightGrad[iRight] += grad;
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        int iLeft = leftValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        int iRight = rightValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        G grad = func(left[iLeft], right[iRight], outputGradValue[iOutput]);
                        lock (rightGrad)
                        {
                            rightGrad[iRight] += grad;
                        }
                    });
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

        private void ExecuteBinaryBackwardLeftAndRight<T, G>(KindBinary kind, Value untypedLeft, Value untypedRight, Value untypedOutput)
            where T : struct, INumber<T>
            where G : struct, IFloatingPointIeee754<G>
        {
            Value<T> leftValue = untypedLeft.As<T>();
            DataBuffer< T> typedLeftData = leftValue.GetOrInitializeDataBuffer();
            T[] left = typedLeftData.GetInitializedDataArray();
            DataBuffer<G> typedGradLeft = leftValue.GetOrInitializeGradBuffer<G>();
            G[] leftGrad = typedGradLeft.GetInitializedDataArray();

            Value<T> rightValue = untypedRight.As<T>();
            DataBuffer< T> typedRightData = rightValue.GetOrInitializeDataBuffer();
            T[] right = typedRightData.GetInitializedDataArray();
            DataBuffer<G> typedGradRight = rightValue.GetOrInitializeGradBuffer<G>();
            G[] rightGrad = typedGradRight.GetInitializedDataArray();

            Value<T> outputValue = untypedOutput.As<T>();
            DataBuffer<G> typedGradOutput = outputValue.GetOrInitializeGradBuffer<G>();
            G[] outputGrad = typedGradOutput.GetInitializedDataArray();

            Func<T, T, G, G> funcLeft = BinaryOperations.GetKindBackwardLeftDelegate<T, G>(kind);
            Func<T, T, G, G> funcRight = BinaryOperations.GetKindBackwardRightDelegate<T, G>(kind);

            int length = outputGrad.Length;
            if (leftValue.Shape == rightValue.Shape)
            {
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        G gradLeft = funcLeft(left[iOutput], right[iOutput], outputGrad[iOutput]);
                        G gradRight = funcRight(left[iOutput], right[iOutput], outputGrad[iOutput]);
                        leftGrad[iOutput] += gradLeft;
                        rightGrad[iOutput] += gradRight;
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        G gradLeft = funcLeft(left[iOutput], right[iOutput], outputGrad[iOutput]);
                        G gradRight = funcRight(left[iOutput], right[iOutput], outputGrad[iOutput]);
                        lock (outputGrad)
                        {
                            leftGrad[iOutput] += gradLeft;
                            rightGrad[iOutput] += gradRight;
                        }
                    });
                }
            }
            else
            {
                if (_parallelOptions.MaxDegreeOfParallelism == 1)
                {
                    for (int iOutput = length - 1; iOutput >= 0; iOutput--)
                    {
                        int iLeft = leftValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        int iRight = rightValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);

                        G gradLeft = funcLeft(left[iLeft], right[iRight], outputGrad[iOutput]);
                        G gradRight = funcRight(left[iLeft], right[iRight], outputGrad[iOutput]);
                        leftGrad[iLeft] += gradLeft;
                        rightGrad[iRight] += gradRight;
                    }
                }
                else
                {
                    Parallel.For(0, length, _parallelOptions, iOutput =>
                    {
                        int iLeft = leftValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);
                        int iRight = rightValue.Shape.GetLinearIndex(iOutput, outputValue.Shape);

                        G gradLeft = funcLeft(left[iLeft], right[iRight], outputGrad[iOutput]);
                        G gradRight = funcRight(left[iLeft], right[iRight], outputGrad[iOutput]);
                        lock (leftGrad)
                        {
                            leftGrad[iLeft] += gradLeft;
                            rightGrad[iRight] += gradRight;
                        }
                    });
                }
            }
        }
    }


}
