using SharpGrad.DifEngine.SyntaxBuilder;
using System.Numerics;

namespace SharpGrad
{
    public interface IValue : IGraphNode<Value>
    {
        IReadOnlyDataBuffer Data { get; }
        IReadOnlyDataBuffer Grad { get; }
        bool IsGradTypeSet { get; }
        DataBuffer<GradType> GetOrInitializeGradBuffer<GradType>() where GradType : struct, IFloatingPointIeee754<GradType>;
        bool IsGradiable { get; set; }
        Value<T> As<T>() where T : struct, INumber<T>;
    }

    public interface IValue<TType> : IValue
        where TType : struct, INumber<TType>
    {
        new IReadOnlyDataBuffer<TType> Data { get; }
    }
}