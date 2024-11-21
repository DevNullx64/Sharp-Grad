using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

namespace SharpGrad.Formula
{
    public class DictionaryList<TKey, TValue> : IDictionary<TKey, TValue>, IReadOnlyList<TValue>
        where TKey : notnull
    {
        protected Dictionary<TKey, TValue> dictionary = [];
        protected List<TValue> values = [];

        public TValue this[TKey key]
        {
            get => dictionary[key];
            set => dictionary[key] = value;
        }

        public TValue this[int index]
        {
            get => values[index];
            set => values[index] = value;
        }

        public ICollection<TKey> Keys => dictionary.Keys;

        public ICollection<TValue> Values => values;

        public int Count => values.Count;

        public bool IsReadOnly => false;

        public void Add(TKey key, TValue value)
        {
            if (dictionary.ContainsKey(key))
                throw new ArgumentException($"An element with the key '{key}' already exists.");

            dictionary.Add(key, value);
            values.Add(value);
        }

        public void Add(KeyValuePair<TKey, TValue> item)
            => Add(item.Key, item.Value);

        public void Clear()
        {
            dictionary.Clear();
            values.Clear();
        }

        public bool Contains(KeyValuePair<TKey, TValue> item)
            => dictionary.TryGetValue(item.Key, out TValue? value)
            && EqualityComparer<TValue>.Default.Equals(value, item.Value);

        public bool ContainsKey(TKey key)
            => dictionary.ContainsKey(key);

        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex)
            => throw new NotSupportedException();

        public bool Remove(TKey key)
        {
            bool result = dictionary.TryGetValue(key, out TValue? value);
            if (result)
            {
                dictionary.Remove(key);
                values.Remove(value!);
            }
            return result;
        }

        public bool Remove(KeyValuePair<TKey, TValue> item)
        {
            bool result = dictionary.TryGetValue(item.Key, out TValue? value)
                && EqualityComparer<TValue>.Default.Equals(value, item.Value);
            if (result)
            {
                dictionary.Remove(item.Key);
                values.Remove(item.Value);
            }
            return result;
        }

        public bool TryGetValue(TKey key, [MaybeNullWhen(false)] out TValue value)
            => dictionary.TryGetValue(key, out value);

        public IEnumerator<TValue> GetEnumerator()
            => values.GetEnumerator();

        IEnumerator<KeyValuePair<TKey, TValue>> IEnumerable<KeyValuePair<TKey, TValue>>.GetEnumerator()
            => dictionary.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator()
            => GetEnumerator();

        public int IndexOf(TValue element)
            => values.IndexOf(element);
        public int IndexOf(TKey key)
            => IndexOf(dictionary[key]);
    }
}