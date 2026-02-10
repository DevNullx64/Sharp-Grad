using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace SharpGrad.DifEngine.SyntaxBuilder.CPU
{
    public partial class DeviceCpu
    {
        ParallelOptions _parallelOptions;
        public int ThreadCount
        {
            get => _parallelOptions.MaxDegreeOfParallelism;
            set => _parallelOptions.MaxDegreeOfParallelism = value;
        }

        public DeviceCpu(int cpuCount)
        {
            _parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = cpuCount };
        }
        public DeviceCpu() :
            this(Environment.ProcessorCount)
        { }


        public bool IsAvailable() => true;

        public string Name => nameof(DeviceCpu);

        public void ResetGradient<TType>(Value<TType> loss)
            where TType : struct, INumber<TType>
        {
            loss.untypedGrad = null;
        }

        /// <summary>
        /// Helper method to find a generic method by name and signature.
        /// </summary>
        /// <param name="methodName">The name of the method to find.</param>
        /// <param name="genericArgCount">The number of generic type parameters.</param>
        /// <param name="parameterTypes">The parameter types of the method (excluding generic parameters).</param>
        /// <returns>The generic MethodInfo, or null if not found.</returns>
        private static MethodInfo? GetGenericMethod(string methodName, int genericArgCount, params Type[] parameterTypes)
        {
            return typeof(DeviceCpu).GetMethods(BindingFlags.NonPublic | BindingFlags.Instance)
                .FirstOrDefault(m => m.Name == methodName &&
                    m.IsGenericMethodDefinition &&
                    m.GetGenericArguments().Length == genericArgCount &&
                    m.GetParameters().Length == parameterTypes.Length &&
                    m.GetParameters().Select(p => p.ParameterType).SequenceEqual(parameterTypes));
        }

        /// <summary>
        /// Helper method to find a generic method definition by name from a specific type.
        /// </summary>
        /// <param name="type">The type to search for the method.</param>
        /// <param name="methodName">The name of the method to find.</param>
        /// <param name="bindingFlags">The binding flags for the search.</param>
        /// <returns>The generic MethodInfo, or null if not found.</returns>
        internal static MethodInfo? FindGenericMethod(Type type, string methodName, BindingFlags bindingFlags = BindingFlags.Public | BindingFlags.Static)
        {
            var methods = type.GetMethods(bindingFlags);
            foreach (var m in methods)
            {
                if (m.Name == methodName && m.IsGenericMethodDefinition)
                {
                    return m;
                }
            }
            return null;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void ParallelFor(int fromInclusive, int toExclusive, Action<Tuple<int , int>> fnc)
        {
            if (_parallelOptions.MaxDegreeOfParallelism == 1)
            {
                fnc(new Tuple<int, int>(fromInclusive, toExclusive));
            }
            else
            {
                Parallel.ForEach(Partitioner.Create(fromInclusive, toExclusive), _parallelOptions, fnc);
            }
        }

        /// <summary>
        /// Fills a span with a neutral element value.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void FillSpan<TType>(Span<TType> output, TType neutralElement)
            where TType : struct, INumber<TType>
        {
            int iOutput = 0;
            if (Vector<TType>.IsSupported && output.Length >= Vector<TType>.Count)
            {
                Vector<TType> neutralVector = new(neutralElement);
                int iVectorEnd = output.Length - (output.Length % Vector<TType>.Count);
                for (; iOutput < iVectorEnd; iOutput += Vector<TType>.Count)
                {
                    neutralVector.CopyTo(output[iOutput..]);
                }
            }
            for (; iOutput < output.Length; iOutput++)
            {
                output[iOutput] = neutralElement;
            }
        }

        /// <summary>
        /// Reduces a broadcasted gradient buffer back to target shape.
        /// Performs iterative reduction until only one dimension remains, then final reduction.
        /// </summary>
        private void ReduceBroadcastedGradientToShape<TGrad>(
            TGrad[] gradBuffer,
            Shape gradBufferShape,
            Shape targetShape,
            Value targetValue)
            where TGrad : struct, IFloatingPointIeee754<TGrad>
        {
            TGrad[] currentBuffer = gradBuffer;
            Shape currentShape = gradBufferShape;

            // Iteratively reduce dimensions that don't exist in targetShape
            for (int iDim = currentShape.Rank - 1; iDim >= 0; iDim--)
            {
                if (currentShape.Rank == targetShape.Rank + 1)
                {
                    // Only one dimension to reduce - stop and handle in final step
                    break;
                }
                
                Dimension dimToReduce = currentShape[iDim];
                if (targetShape.IndexOf(dimToReduce) < 0)
                {
                    Shape reducedShape = currentShape.Remove(dimToReduce);
                    TGrad[] reducedBuffer = new TGrad[reducedShape.Size];
                    int strideToReduce = currentShape.GetStride(iDim);

                    ParallelFor(0, reducedBuffer.Length, range =>
                    {
                        Span<TGrad> toReduceSpan = currentBuffer;
                        Span<TGrad> reducedSpan = reducedBuffer;
                        
                        for (int iReduced = range.Item1; iReduced < range.Item2; iReduced++)
                        {
                            int iToReduce = reducedShape.GetLinearIndex(iReduced, currentShape, false);
                            
                            TGrad sum = TGrad.Zero;
                            for (int j = 0; j < dimToReduce.Size; j++)
                            {
                                sum += toReduceSpan[iToReduce + j * strideToReduce];
                            }
                            reducedSpan[iReduced] = sum;
                        }
                    });
                    
                    currentBuffer = reducedBuffer;
                    currentShape = reducedShape;
                }
            }

            // Final reduction: reduce the last remaining dimension directly into target gradient
            int iDimToReduce;
            for (iDimToReduce = currentShape.Rank - 1; iDimToReduce >= 0; iDimToReduce--)
            {
                Dimension dim = currentShape[iDimToReduce];
                if (targetShape.IndexOf(dim) < 0)
                {
                    break;
                }
            }
            
            Dimension dimToReduceFinal = currentShape[iDimToReduce];
            int strideToReduceFinal = currentShape.GetStride(iDimToReduce);
            int dimSizeFinal = dimToReduceFinal.Size;

            ParallelFor(0, targetShape.Size, range =>
            {
                Span<TGrad> targetGrad = targetValue.GetInitializedGrad<TGrad>();
                Span<TGrad> bufferSpan = currentBuffer;

                for (int iTarget = range.Item1; iTarget < range.Item2; iTarget++)
                {
                    int iToReduce = targetShape.GetLinearIndex(iTarget, currentShape, false);

                    TGrad sum = TGrad.Zero;
                    for (int j = 0; j < dimSizeFinal; j++)
                    {
                        sum += bufferSpan[iToReduce + j * strideToReduceFinal];
                    }
                    targetGrad[iTarget] += sum;
                }
            });
        }
    }
}
