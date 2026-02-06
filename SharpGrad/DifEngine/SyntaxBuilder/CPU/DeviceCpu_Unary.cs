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
        // Cache for the ExecuteUnaryForward delegates
        private readonly Dictionary<
            Type,
            Action<Value, Value>> cacheExecuteUnaryForwards = [];

        /// <summary>
        /// Executes the forward pass of a unary operation on the given input and stores the result in the given output.
        /// </summary>
        /// <param name="input">The input Value.</param>
        /// <param name="output">The output Value.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the forward pass for different element types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="DeviceCpu.ExecuteUnaryForward"/> method is not found.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteUnaryForward(Value input, Value output)
        {
            // Check if a cached delegate exists for the input element type
            Type inputElement = input.ElementType;
            if (!cacheExecuteUnaryForwards.TryGetValue(inputElement, out Action<Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteUnaryForward<T>(Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteUnaryForward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteUnaryForward)} not found.");
                method = method.MakeGenericMethod(inputElement);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteUnaryForwards[inputElement] = action;
            }
            // Call the generic ExecuteUnaryForward<T> method
            action(input, output);
        }

        /// <summary>
        /// Executes the forward pass of a unary operation for a given input and stores the result in the given output.
        /// </summary>
        /// <typeparam name="TType">The type of the elements in the input and output Values. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <param name="untypedInput">The input <see cref="Value"/>.</param>
        /// <param name="untypedOutput">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// <paramref name="untypedInput"/> and <paramref name="untypedOutput"/> must be of type <see cref="Value{T}"/>.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input or output are not of type <see cref="Value{T}"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="UnaryOperations"/>."[<see cref="Value.Kind"/>]Forward" method is not found.</exception>
        private void ExecuteUnaryForward<TType>(Value untypedInput, Value untypedOutput)
            where TType : struct, INumber<TType>
        {
            TType[] input = untypedInput.GetInitializedData<TType>();
            TType[] output = untypedOutput.GetOrInitializeData<TType>();

            // Get the appropriate method for the unary operation
            Func<TType, TType> operation = UnaryOperations.GetKindForwardDelegate<TType>(untypedOutput.Kind);

            // Perform the unary operation
            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = input.Length - 1; iOutput >= 0; iOutput--)
                {
                    output[iOutput] = operation(input[iOutput]);
                }
            }
            else
            {
                Parallel.For(0, input.Length, _parallelOptions, iOutput =>
                {
                    output[iOutput] = operation(input[iOutput]);
                });
            }
        }

        // Cache for the ExecuteCastForward delegates
        private readonly Dictionary<
            (Type From, Type To),
            Action<Value, Value>> cacheExecuteCastForwards = [];

        /// <summary>
        /// Executes the forward pass of a cast operation from input to output.
        /// </summary>
        /// <param name="input">The input <see cref="Value"/>.</param>
        /// <param name="output">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the cast operation for different combinations of source and target types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="DeviceCpu.ExecuteCastForward"/> method is not found.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteCastForward(Value input, Value output)
        {
            // Check if a cached delegate exists for the input and output element types
            (Type From, Type To) key = (input.ElementType, output.ElementType);
            if (!cacheExecuteCastForwards.TryGetValue(key, out Action<Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteCastForward<TFrom, TTo>(Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteCastForward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteCastForward)} not found.");
                method = method.MakeGenericMethod(key.From, key.To);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteCastForwards[key] = action;
            }
            // Call the generic CastForward<TFrom, TTo> method
            action(input, output);
        }

        /// <summary>
        /// Executes the forward pass of a cast operation from type TFrom to type TTo.
        /// </summary>
        /// <typeparam name="TFrom">The source type of the cast operation. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <typeparam name="TTo">The target type of the cast operation. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <param name="untypedInput">The input <see cref="Value"/>.</param>
        /// <param name="untypedOutput">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// <paramref name="untypedInput"/> must be of type <see cref="Value{TFrom}"/> and <paramref name="untypedOutput"/> must be of type <see cref="Value{TTo}"/>.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input and output are not of the expected types.</exception>
        private void ExecuteCastForward<TFrom, TTo>(Value untypedInput, Value untypedOutput)
            where TFrom : struct, INumber<TFrom>
            where TTo : struct, INumber<TTo>
        {
            TFrom[] input = untypedInput.GetInitializedData<TFrom>();
            TTo[] output = untypedOutput.GetOrInitializeData<TTo>();

            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = output.Length - 1; iOutput >= 0; iOutput--)
                {
                    output[iOutput] = UnaryOperations.CastForward<TFrom, TTo>(input[iOutput]);
                }
            }
            else
            {
                Parallel.For(0, output.Length, _parallelOptions, iOutput =>
                {
                    output[iOutput] = UnaryOperations.CastForward<TFrom, TTo>(input[iOutput]);
                });
            }
        }

        // Cache for the ExecuteUnaryBackward delegates
        private readonly Dictionary<
            (Type Value, Type Gradient),
            Action<Value, Value>> cacheExecuteUnaryBackwards = [];

        /// <summary>
        /// Executes the backward pass of a unary operation on the given input and output Values.
        /// </summary>
        /// <param name="input">The input <see cref="Value"/>.</param>
        /// <param name="output">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the backward pass for different combinations of element types and gradient types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the ExecuteUnaryBackward method is not found or if the input and output types are incompatible.</exception>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ExecuteUnaryBackward(Value input, Value output)
        {
            // Call the generic ExecuteUnaryBackward<T, G>(Value, Value) method
            (Type Value, Type Gradient) key = (output.ElementType, output.Grad.ElementType);
            if (!cacheExecuteUnaryBackwards.TryGetValue(key, out Action<Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteUnaryBackward<T, G>(Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteUnaryBackward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteUnaryBackward)} not found.");
                method = method.MakeGenericMethod(key.Value, key.Gradient);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate<Action<Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteUnaryBackwards[key] = action;
            }
            action(input, output);
        }

        /// <summary>
        /// Executes the backward pass of a unary operation on the given input and output Values.
        /// </summary>
        /// <typeparam name="TType">The type of the elements in the input and output Values. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <typeparam name="TGrad">The type of the gradients. Must be a struct that implements <see cref="IFloatingPointIeee754{G}"/>.</typeparam>
        /// <param name="input">The input <see cref="Value"/>.</param>
        /// <param name="output">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the backward pass for different combinations of element types and gradient types.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input or output are not of type <see cref="Value{T}"/>.</exception>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="UnaryOperations"/>."[<see cref="Value.Kind"/>]Backward" method is not found.</exception>
        private void ExecuteUnaryBackward<TType, TGrad>(Value untypedInput, Value untypedOutput)
            where TType : struct, INumber<TType>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            TType[] input = untypedInput.GetInitializedData<TType>();
            TGrad[] gradInput = untypedInput.GetOrInitializeGrad<TGrad>();
            TType[] output = untypedOutput.GetInitializedData<TType>();
            TGrad[] gradOutput = untypedOutput.GetInitializedGrad<TGrad>();

            // Create a delegate for the method
            Func<TType, TType, TGrad, TGrad> func = UnaryOperations.GetKindBackwardDelegate<TType, TGrad>(untypedOutput.Kind);

            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = gradOutput.Length - 1; iOutput >= 0; iOutput--)
                {
                    gradInput[iOutput] += func(input[iOutput], output[iOutput], gradOutput[iOutput]);
                }
            }
            else
            {
                Parallel.For(0, gradOutput.Length, _parallelOptions, iOutput =>
                {
                    gradInput[iOutput] += func(input[iOutput], output[iOutput], gradOutput[iOutput]);
                });
            }
        }

        // Cache for the ExecuteCastBackward delegates
        private readonly Dictionary<
            (Type From, Type To, Type Grad),
            Action<Value, Value>> cacheExecuteCastBackwards = [];

        /// <summary>
        /// Executes the backward pass of a cast operation from output to input.
        /// </summary>
        /// <param name="input">The input <see cref="Value"/>.</param>
        /// <param name="output">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// This method uses caching to optimize the execution of the cast operation for different combinations of source and target types and gradient types.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if the <see cref="DeviceCpu.ExecuteCastBackward"/> method is not found.</exception>
        private void ExecuteCastBackward(Value input, Value output)
        {
            (Type From, Type To, Type Grad) key = (input.ElementType, output.ElementType, output.Grad.ElementType);
            if (!cacheExecuteCastBackwards.TryGetValue(key, out Action<Value, Value>? action))
            {
                // Get the MethodInfo for the instance method ExecuteCastBackward<From, To, Grad>(Value, Value)
                MethodInfo method = typeof(DeviceCpu).GetMethod(
                    nameof(ExecuteCastBackward),
                    BindingFlags.NonPublic | BindingFlags.Instance,
                    [typeof(Value), typeof(Value)]
                ) ?? throw new InvalidOperationException($"Method {nameof(DeviceCpu)}.{nameof(ExecuteCastBackward)} not found.");
                method = method.MakeGenericMethod(key.From, key.To, key.Grad);

                // Create a delegate for the method using the current instance
                action = method.CreateDelegate< Action<Value, Value>>(this);

                // Cache the delegate for future use
                cacheExecuteCastBackwards[key] = action;
            }
            // Call the generic CastBackward<TFrom, TTo, G> method
            action(input, output);
        }

        /// <summary>
        /// Executes the backward pass of a cast operation from type TTo to type TFrom.
        /// </summary>
        /// <typeparam name="TFrom">The source type of the cast operation. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <typeparam name="TTo">The target type of the cast operation. Must be a struct that implements <see cref="INumber{T}"/>.</typeparam>
        /// <typeparam name="TGrad">The type of the gradients. Must be a struct that implements <see cref="IFloatingPointIeee754{G}"/>.</typeparam>
        /// <param name="untypedInput">The input <see cref="Value"/>.</param>
        /// <param name="untypedOutput">The output <see cref="Value"/>.</param>
        /// <remarks>
        /// <paramref name="untypedInput"/> must be of type <see cref="Value{TFrom}"/> and <paramref name="untypedOutput"/> must be of type <see cref="Value{TTo}"/>.
        /// </remarks>
        /// <exception cref="InvalidCastException">Thrown if the input and output are not of the expected types.</exception>
        private void ExecuteCastBackward<TFrom, TTo, TGrad>(Value untypedInput, Value untypedOutput)
            where TFrom : struct, INumber<TFrom>
            where TTo : struct, INumber<TTo>
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            if (!untypedInput.IsGradiable)
                return;

            TFrom[] input = untypedInput.GetInitializedData<TFrom>();
            TGrad[] inputGrad = untypedInput.GetOrInitializeGrad<TGrad>();
            TTo[] output = untypedOutput.GetInitializedData<TTo>();
            TGrad[] outputGrad = untypedOutput.GetInitializedGrad<TGrad>();

            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                for (int iOutput = input.Length - 1; iOutput >= 0; iOutput--)
                {
                    inputGrad[iOutput] += UnaryOperations.CastBackward(input[iOutput], output[iOutput], outputGrad[iOutput]);
                }
            }
            else
            {
                Parallel.For(0, input.Length, _parallelOptions, iOutput =>
                {
                    inputGrad[iOutput] += UnaryOperations.CastBackward(input[iOutput], output[iOutput], outputGrad[iOutput]);
                });
            }
        }
    }
}