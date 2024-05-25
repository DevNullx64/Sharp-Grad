namespace SharpGrad.Tensors
{
    public interface ITensor {
        Shape Shape { get; }
    }

    public interface ITensor<TType> : ITensor
    {
        TType this[params int[] indices] { get; set; }
    }
}
