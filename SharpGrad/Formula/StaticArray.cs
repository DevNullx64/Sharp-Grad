using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Formula
{
    public abstract class StaticArray<T> : IReadOnlyList<T>
    {
        public abstract T this[int index] { get; set; }

        public abstract int Count { get; }

        public abstract IEnumerator<T> GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator()
            => GetEnumerator();
    }

    internal class StaticArray1<T>(T item = default)
        : StaticArray<T>
    {
        public T Item1 = item;
        public override T this[int index]
        {
            get => index switch
            {
                1 => Item1,
                _ => throw new IndexOutOfRangeException()
            };
            set
            {
                switch (index)
                {
                    case 1:
                        Item1 = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public override int Count => 1;

        public override IEnumerator<T> GetEnumerator()
        {
            yield return Item1;
        }
    }

    internal class StaticArray2<T>(T item1 = default, T item2 = default)
        : StaticArray<T>
    {
        public T Item1 = item1;
        public T Item2 = item2;
        public override T this[int index]
        {
            get => index switch
            {
                1 => Item1,
                2 => Item2,
                _ => throw new IndexOutOfRangeException()
            };
            set
            {
                switch (index)
                {
                    case 1:
                        Item1 = value;
                        break;
                    case 2:
                        Item2 = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public override int Count => 2;

        public override IEnumerator<T> GetEnumerator()
        {
            yield return Item1;
            yield return Item2;
        }
    }

    internal class StaticArray3<T>(T item1 = default, T item2 = default, T item3 = default)
        : StaticArray<T>
    {
        public T Item1 = item1;
        public T Item2 = item2;
        public T Item3 = item3;
        public override T this[int index]
        {
            get => index switch
            {
                1 => Item1,
                2 => Item2,
                3 => Item3,
                _ => throw new IndexOutOfRangeException()
            };
            set
            {
                switch (index)
                {
                    case 1:
                        Item1 = value;
                        break;
                    case 2:
                        Item2 = value;
                        break;
                    case 3:
                        Item3 = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public override int Count => 3;

        public override IEnumerator<T> GetEnumerator()
        {
            yield return Item1;
            yield return Item2;
            yield return Item3;
        }
    }

    internal class StaticArray4<T>(T item1 = default, T item2 = default, T item3 = default, T item4 = default)
        : StaticArray<T>
    {
        public T Item1 = item1;
        public T Item2 = item2;
        public T Item3 = item3;
        public T Item4 = item4;
        public override T this[int index]
        {
            get => index switch
            {
                1 => Item1,
                2 => Item2,
                3 => Item3,
                4 => Item4,
                _ => throw new IndexOutOfRangeException()
            };
            set
            {
                switch (index)
                {
                    case 1:
                        Item1 = value;
                        break;
                    case 2:
                        Item2 = value;
                        break;
                    case 3:
                        Item3 = value;
                        break;
                    case 4:
                        Item4 = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public override int Count => 4;

        public override IEnumerator<T> GetEnumerator()
        {
            yield return Item1;
            yield return Item2;
            yield return Item3;
            yield return Item4;
        }
    }

    internal class StaticArray5<T>(T item1 = default, T item2 = default, T item3 = default, T item4 = default, T item5 = default)
        : StaticArray<T>
    {
        public T Item1 = item1;
        public T Item2 = item2;
        public T Item3 = item3;
        public T Item4 = item4;
        public T Item5 = item5;
        public override T this[int index]
        {
            get => index switch
            {
                1 => Item1,
                2 => Item2,
                3 => Item3,
                4 => Item4,
                5 => Item5,
                _ => throw new IndexOutOfRangeException()
            };
            set
            {
                switch (index)
                {
                    case 1:
                        Item1 = value;
                        break;
                    case 2:
                        Item2 = value;
                        break;
                    case 3:
                        Item3 = value;
                        break;
                    case 4:
                        Item4 = value;
                        break;
                    case 5:
                        Item5 = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public override int Count => 5;

        public override IEnumerator<T> GetEnumerator()
        {
            yield return Item1;
            yield return Item2;
            yield return Item3;
            yield return Item4;
            yield return Item5;
        }
    }

    internal class StaticArray6<T>(T item1 = default, T item2 = default, T item3 = default, T item4 = default, T item5 = default, T item6 = default)
        : StaticArray<T>
    {
        public T Item1 = item1;
        public T Item2 = item2;
        public T Item3 = item3;
        public T Item4 = item4;
        public T Item5 = item5;
        public T Item6 = item6;
        public override T this[int index]
        {
            get => index switch
            {
                1 => Item1,
                2 => Item2,
                3 => Item3,
                4 => Item4,
                5 => Item5,
                6 => Item6,
                _ => throw new IndexOutOfRangeException()
            };
            set
            {
                switch (index)
                {
                    case 1:
                        Item1 = value;
                        break;
                    case 2:
                        Item2 = value;
                        break;
                    case 3:
                        Item3 = value;
                        break;
                    case 4:
                        Item4 = value;
                        break;
                    case 5:
                        Item5 = value;
                        break;
                    case 6:
                        Item6 = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public override int Count => 6;

        public override IEnumerator<T> GetEnumerator()
        {
            yield return Item1;
            yield return Item2;
            yield return Item3;
            yield return Item4;
            yield return Item5;
            yield return Item6;
        }
    }

    internal class StaticArray7<T>(T item1 = default, T item2 = default, T item3 = default, T item4 = default, T item5 = default, T item6 = default, T item7 = default)
        : StaticArray<T>
    {
        public T Item1 = item1;
        public T Item2 = item2;
        public T Item3 = item3;
        public T Item4 = item4;
        public T Item5 = item5;
        public T Item6 = item6;
        public T Item7 = item7;
        public override T this[int index]
        {
            get => index switch
            {
                1 => Item1,
                2 => Item2,
                3 => Item3,
                4 => Item4,
                5 => Item5,
                6 => Item6,
                7 => Item7,
                _ => throw new IndexOutOfRangeException()
            };
            set
            {
                switch (index)
                {
                    case 1:
                        Item1 = value;
                        break;
                    case 2:
                        Item2 = value;
                        break;
                    case 3:
                        Item3 = value;
                        break;
                    case 4:
                        Item4 = value;
                        break;
                    case 5:
                        Item5 = value;
                        break;
                    case 6:
                        Item6 = value;
                        break;
                    case 7:
                        Item7 = value;
                        break;
                    default:
                        throw new IndexOutOfRangeException();
                }
            }
        }

        public override int Count => 7;

        public override IEnumerator<T> GetEnumerator()
        {
            yield return Item1;
            yield return Item2;
            yield return Item3;
            yield return Item4;
            yield return Item5;
            yield return Item6;
            yield return Item7;
        }
    }
}
