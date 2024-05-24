using ILGPU;
using ILGPU.Runtime;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Formats.Tar;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors
{
    internal static class Tensors
    {
        public static Dictionary<ITensor, long> Instances = [];

        private static Context GetContext()
        {
            Context result = Context.Create(builder => builder.AllAccelerators());
            Debug.WriteLine($"Context created: {result}");
            return result;
        }
        private static readonly Context context = GetContext();

        private static Device GetDevice(Context context)
        {
            Device result = context.GetPreferredDevice(preferCPU: false);
            Debug.WriteLine($"Device created: {result}");
            return result;
        }
        private static readonly Device device = GetDevice(context);
        public static readonly Accelerator Accelerator = device.CreateAccelerator(context);

    }

    public interface ITensor {
        bool IsOnGpu { get; }
        Shape Shape { get; }
    }

    public interface ITensor<TType>: ITensor
    {
        TType this[params int[] indices] { get; set; }
    }
    public readonly partial struct Tensor<TType>: ITensor<TType>
        where TType : unmanaged, IFloatingPoint<TType>
    {
        private readonly TType[] data;
        private readonly TType[]? gradients;
        private readonly Shape shape;
        public Shape Shape => shape;

        public bool IsOnGpu => false;

        public readonly TType this[params int[] indices]
        {
            get => data[shape.GetFlattenedIndex(indices)];
            set => data[shape.GetFlattenedIndex(indices)] = value;
        }


        public Tensor(Shape shape, TType[] data)
        {
            if (data.Length != shape.Aggregate(1, (a, b) => a * b))
                throw new ArgumentException($"Expected data length {shape.Aggregate(1, (a, b) => a * b)}, got {data.Length}");

            this.data = data;
            gradients = new TType[data.Length];
            this.shape = shape;
            Tensors.Instances.Add(this, DateTime.Now.Ticks);
        }
        public Tensor(Shape shape) : this(shape, new TType[shape.Aggregate(1, (a, b) => a * b)]) { }
        public Tensor(params Dim[] shape) : this(new Shape(shape)) { }

        public readonly void AddGradient(Tensor<TType> gradient)
        {
            if(gradient.shape != shape)
                throw new ArgumentException($"Expected gradient shape {shape}, got {gradient.shape}");

            if (gradients != null)
            {
                for (int i = 0; i < gradients.Length; i++)
                    gradients[i] += gradient.data[i];
            }
        }
    }
}
