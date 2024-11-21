using SharpGrad.Memory;
using System;

namespace SharpGrad.Tensors.Formula
{
    /// <summary>
    /// Represents the result of a computation.
    /// </summary>
    /// <typeparam name="TResult">The type of the result.</typeparam>
    public class Result<TResult>
        where TResult : unmanaged
    {
        public readonly Shape Shape;
        internal AcceleratorBuffer<TResult>? Content { get; private set; }
        internal AcceleratorBuffer<TResult>? Gradient { get; private set; }

        public readonly bool IsComputable;

        private bool isComputed;
        public bool IsComputed
        {
            get => IsComputable && isComputed;
            set => isComputed = value && IsComputable;
        }

        public bool HasContent
        {
            get => Content is not null;
            set
            {
                if (HasContent != value)
                {
                    if (value)
                    {
                        Content = AcceleratorExtender.DefaultExtender.Allocate<TResult>(Shape.Length);
                        isComputed = false;
                    }
                    else
                    {
                        if (Content is not null)
                        {
                            Content.Dispose();
                            Content = null;
                        }
                        IsGradiable = false;
                    }
                }
            }
        }

        public TResult[] GetContent()
            => Content?.SafeCPUData ?? throw new InvalidOperationException("The result is not initialized.");

        public bool IsGradiable
        {
            get => Gradient is not null;
            set
            {
                if (IsGradiable != value)
                {
                    if (value)
                    {
                        HasContent = true;
                        Gradient = AcceleratorExtender.DefaultExtender.Allocate<TResult>(Shape.Length);
                    }
                    else
                    {
                        Gradient = null;
                    }
                }
            }
        }


        public TResult[] GetGradients()
            => Gradient?.SafeCPUData ?? throw new InvalidOperationException("The gradient is not initialized.");

        private Result(Shape shape, bool isGradiable, AcceleratorBuffer<TResult>? content, bool isComputable)
        {
            Shape = shape;
            Content = content;
            IsGradiable = isGradiable;
            IsComputable = isComputable;
        }

        internal Result(Shape shape, bool isGradiable, bool init, bool isComputable)
            : this(shape, isGradiable, init ? AcceleratorExtender.DefaultExtender.Allocate<TResult>(shape.Length) : null, isComputable)
        { }

        internal void ResetGradient()
            => Gradient?.Reset();
    }
}