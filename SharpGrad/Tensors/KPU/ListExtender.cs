using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpGrad.Tensors.KPU
{
    public static class ListExtender
    {
        /// <summary>
        /// Inserts an item into the list at the first available null index. Add the item to the end of the list if no null index is found.
        /// </summary>
        /// <typeparam name="T">The type of the list.</typeparam>
        /// <param name="this">The list to insert the item into.</param>
        /// <param name="item">The item to insert.</param>
        /// <returns>The index of the inserted item.</returns>
        public static int Insert<T>(this List<T?> @this, T item)
        {
            int count = @this.Count;
            for (int i = 0; i < count; i++)
                if (@this[i] is null)
                {
                    @this[i] = item;
                    return i;
                }
            @this.Add(item);
            return count;
        }
    }
}
