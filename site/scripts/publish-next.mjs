// Publish the post at the top of the approved queue: status -> published, date -> today (UTC), queue renumbered.
// It only edits the file; commit, push and deploy/ship.sh are separate steps.
//   node scripts/publish-next.mjs           publish the next post
//   node scripts/publish-next.mjs --dry     show what would publish
import * as fm from '../queue/fm.mjs';

const [next, ...rest] = fm.approved();
if (!next) { console.error('queue is empty: approve a post at /queue/ first'); process.exit(1); }
const today = new Date().toISOString().slice(0, 10);
if (process.argv.includes('--dry')) { console.log(`next: ${next.slug} (${next.title})`); process.exit(0); }
fm.update(next.slug, { status: 'published', queue: null, date: today });
fm.reorder(rest.map((p) => p.slug));
console.log(`published ${next.slug} dated ${today}; ${rest.length} left in the queue`);
