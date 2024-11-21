using System;

namespace SharpGrad.Tensors.KPU
{
    public abstract class LinearArray<T>(Array array)
        where T : unmanaged
    {
        protected Array array = array;
        protected abstract int[] GetIndices(int index);
        protected int[] GetIndices(Index index)
            => GetIndices(index.IsFromEnd ? array.Length - index.Value : index.Value);

        public T this[Index index]
        {
            get => (T)array.GetValue(GetIndices(index));
            set => array.SetValue(value, GetIndices(index));
        }
    }

    public class LinearArray1<T>(T[] array) : LinearArray<T>(array)
        where T : unmanaged
    {
        protected override int[] GetIndices(int index)
            => [index];
    }

    public class LinearArray2<T>(T[,] array) : LinearArray<T>(array)
        where T : unmanaged
    {
        protected override int[] GetIndices(int index)
            => [index / array.GetLength(1), index % array.GetLength(1)];
    }

    public class LinearArrayN<T> : LinearArray<T>
        where T : unmanaged
    {
        public LinearArrayN(T[,,] array) : base(array) { }
        public LinearArrayN(T[,,,] array) : base(array) { }
        public LinearArrayN(T[,,,,] array) : base(array) { }
        public LinearArrayN(T[,,,,,] array) : base(array) { }
        public LinearArrayN(T[,,,,,,] array) : base(array) { }

        protected override int[] GetIndices(int index)
        {
            int product = 1;
            int[] indices = new int[array.Rank];

            for (int i = array.Rank - 1; i >= 0; i--)
            {
                indices[i] = index / product;
                index %= product;
                product *= array.GetLength(i);
            }

            return indices;
        }
    }
}
