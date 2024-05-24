namespace SharpGrad.Tensors
{
    public struct Operation
    {
        public readonly OpCode OpCode;
        public int Left;
        public int Right;
        public int Result;
    }
}