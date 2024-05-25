using System;
using System.Collections.Generic;
using System.Formats.Tar;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors
{
    public partial class Tensor<TType>: ITensor<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        private readonly DeviceBuffer<TType> data_;
        private readonly TType[]? gradients;
        private readonly Shape shape;
        public Shape Shape => shape;

        public bool IsOnGpu => false;

        public TType this[params int[] indices]
        {
            get => data_.CPUData[shape.GetFlattenedIndex(indices)];
            set => data_.CPUData[shape.GetFlattenedIndex(indices)] = value;
        }


        public Tensor(Shape shape, TType[] data)
        {
            if (data.Length != shape.Aggregate(1, (a, b) => a * b))
                throw new ArgumentException($"Expected data length {shape.Aggregate(1, (a, b) => a * b)}, got {data.Length}");

            var deviceBuffer = new DeviceBuffer<TType>(data.Length);
            data_ = deviceBuffer;
            gradients = new TType[data.Length];
            this.shape = shape;
        }
        public Tensor(Shape shape) : this(shape, new TType[shape.Aggregate(1, (a, b) => a * b)]) { }
        public Tensor(params Dim[] shape) : this(new Shape(shape)) { }

        public void AddGradient(Tensor<TType> gradient)
        {
            if(gradient.shape != shape)
                throw new ArgumentException($"Expected gradient shape {shape}, got {gradient.shape}");

            if (gradients != null)
            {
                for (int i = 0; i < gradients.Length; i++)
                    gradients[i] += gradient.data_.CPUData[i];
            }
        }
    }
}
